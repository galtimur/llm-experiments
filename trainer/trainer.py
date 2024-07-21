import random
import shutil
from math import ceil
from pathlib import Path

import torch
import torch.nn.functional as tofu
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (  # StoppingCriteria,; StoppingCriteriaList,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from data.s3_data_exchange import upload_directory_s3


class Trainer:
    def __init__(
        self,
        config,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        perform_sanity_check: bool = True,
    ):
        self.train_dataloader = train_data_loader
        self.val_dataloader = val_data_loader
        self.sanity_check_complete = False

        self.config = config
        self.train_args = config.train
        self.model_args = config.model
        self.general_args = self.config.general
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        random_id = random.randint(1, 100000)
        self.wand_run_name = f"{self.model_args.name[:6]}_len-{self.model_args.sequence_length}_bs-{self.train_args.train_batch_size}_lr-{self.train_args.max_lr}--id{random_id}"

        self.model, self.tokenizer = self.get_model()

        self.scheduler_mapping = {
            "linear": get_linear_schedule_with_warmup,
            "cosine": get_cosine_schedule_with_warmup,
        }

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.train_args.max_lr),
            weight_decay=self.train_args.weight_decay,
        )

        self.accum_steps = 1
        self.step_args = self._compute_steps()
        self.scheduler = self.define_scheduler()

        self.batches_done = 0
        self.mini_batches_done = 0
        self.loss_acc = 0.0

        if perform_sanity_check:
            self.sanity_check()

    def get_model(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        if self.train_args.precision == "fp32":
            dtype = torch.float32
        else:
            dtype = torch.bfloat16
        if self.model_args.use_flash_attn:
            flash_attn = "flash_attention_2"
        else:
            flash_attn = None
        if self.model_args.pretrained:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_args.name,
                torch_dtype=dtype,
                attn_implementation=flash_attn,
            )
        else:
            config_model = AutoConfig.from_pretrained(self.model_args.name)
            model = AutoModelForCausalLM.from_config(
                config_model, torch_dtype=dtype, attn_implementation=flash_attn
            )
        tokenizer = AutoTokenizer.from_pretrained(self.model_args.name)
        model.to(self.device)
        model.train()

        return model, tokenizer

    def _compute_steps(self):
        # steps|samples counters
        # Prioritization: samples-steps-ratio
        # samples = steps*accum_batch_size
        step_args = self.train_args.steps
        accum_steps = (
            self.train_args.train_batch_size // self.train_args.train_mini_batch_size
        )
        self.train_args.accum_steps = accum_steps
        full_batch_size = accum_steps * self.train_args.train_mini_batch_size
        if full_batch_size != self.train_args.train_batch_size:
            print("! Global batch size is not devisable by mini !")
            print(f"! New global batch size = {full_batch_size}")
            self.train_args.train_batch_size = full_batch_size

        if step_args.warmup_samples is not None:
            step_args.warmup_steps_or_ratio = (
                step_args.warmup_samples // full_batch_size
            )
        if step_args.val_every_sample is not None:
            step_args.val_every_step = step_args.val_every_sample // full_batch_size
        if step_args.save_every_sample is not None:
            step_args.save_every_step = step_args.save_every_sample // full_batch_size
        if step_args.max_train_samples is not None:
            step_args.max_train_steps = step_args.max_train_samples // full_batch_size
        elif step_args.max_train_steps is None:
            step_args.max_train_steps = (
                step_args.num_epochs
                * len(self.train_dataloader)
                * self.train_args.train_mini_batch_size
                // full_batch_size
            )
        if step_args.max_val_samples is not None:
            step_args.max_val_steps = (
                step_args.max_val_samples // self.train_args.val_batch_size
            )
        elif step_args.max_val_steps is None:
            step_args.max_val_steps = len(self.val_dataloader)
        step_args.warmup_steps = self._compute_warmup(
            step_args.warmup_steps_or_ratio, step_args.max_train_steps
        )
        self.train_args.steps = step_args
        self.accum_steps = accum_steps

        return step_args

    @staticmethod
    def _compute_warmup(steps_or_ratio: int | float, total_steps: int) -> int:
        if isinstance(steps_or_ratio, int):
            return steps_or_ratio
        elif isinstance(steps_or_ratio, float):
            return ceil(steps_or_ratio * total_steps)
        else:
            raise ValueError(
                f"Invalid value for `warmup_steps`, got {type(steps_or_ratio)}"
            )

    def define_scheduler(self):
        scheduler_fetcher = self.scheduler_mapping[self.train_args.scheduler]
        return scheduler_fetcher(
            optimizer=self.opt,
            num_warmup_steps=self.step_args.warmup_steps,
            num_training_steps=self.step_args.max_train_steps,
        )

    def run_epoch(self) -> None:
        if not self.sanity_check_complete:
            self.validation(limit=20)
            self._save_ckpt()
            self.sanity_check_complete = True

        for batch_idx, batch in tqdm(
            enumerate(self.train_dataloader, start=1),
            total=len(self.step_args.max_train_steps),
        ):
            to_log = self.training_step(batch, batch_idx)
            if to_log is not None:
                wandb.log(to_log)

            if (to_log is not None) and (
                self.batches_done % self.step_args.val_every_step == 0
            ):
                print(f"validation on step {self.batches_done}")
                to_log = self.validation()
                wandb.log(to_log)

            if (to_log is not None) and (
                self.batches_done % self.step_args.save_every_step == 0
            ):
                print(f"checkpoint on step: {self.batches_done}")
                self._save_ckpt()

        to_log = self.validation()
        wandb.log(to_log)
        self._save_ckpt()

    def run_training(self) -> None:
        total_epochs = self.config["num_epochs"]
        for epoch_id in range(total_epochs):
            print(f"Epoch {epoch_id} / {total_epochs} start")
            self.run_epoch()

    def validation(self, limit: int = -1) -> dict[str, float]:
        self.model.eval()
        full_val_loss = 0
        if limit < 0:
            limit = len(self.val_dataloader)
        for batch_idx, batch in enumerate(self.val_dataloader, start=1):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model_step(batch, return_dict=True)
                full_val_loss += outputs["loss"].item()

            if batch_idx > limit:
                break

        val_loss_mean = full_val_loss / batch_idx
        to_log = {
            "val/loss": val_loss_mean,
        }
        self.model.train()
        # to_log.update(log_batch)

        return to_log

    def model_step(self, batch, return_dict: bool = False):
        outputs = self.model(**batch, return_dict=True)
        targets = batch["input_ids"][:, 1:]
        logits = outputs["logits"][:, :-1]
        # Is it neccesary?
        logits = logits.to(torch.float32)
        loss_mask = batch["attention_mask"][:, 1:]
        loss_tensor = tofu.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        )

        loss_tensor = loss_mask * loss_tensor.reshape_as(loss_mask)
        loss = loss_tensor.sum() / loss_mask.sum()

        if return_dict:
            return {"loss": loss, "logits": logits, "loss_tensor": loss_tensor}

        return loss

    def training_step(self, batch, batch_idx: int) -> None | dict:

        to_log = {}
        batch = {k: v.to(self.device) for k, v in batch.items()}

        loss = self.model_step(batch) / self.accum_steps
        loss.backward()

        self.loss_acc += loss.item()
        if (batch_idx % self.accum_steps) == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_args.max_norm, norm_type=2.0
            )
            self.opt.step()
            self.opt.zero_grad()
            lr = self.opt.param_groups[0]["lr"]
            self.scheduler.step()

            to_log["train/loss"] = self.loss_acc
            to_log["lr"] = lr

            self.batches_done += 1
            self.loss_acc = 0.0
            to_log["grad_norm"] = grad_norm
            to_log["samples"] = batch_idx * self.train_args.train_mini_batch_size
            return to_log

        self.mini_batches_done += 1
        return None

    def _save_ckpt(self) -> None:
        pth = self.general_args.checkpoints_path
        local_path = Path(f"{pth}/{self.wand_run_name}-{self.batches_done}")
        local_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(local_path))
        self.tokenizer.save_pretrained(str(local_path))

        if self.general_args.upload_to_s3:
            s3_path = self.general_args.s3_checkpoint_path
            attempts = 0
            while attempts < 3:
                try:
                    upload_directory_s3(
                        local_path,
                        f"{s3_path}/{self.wand_run_name}-{self.batches_done}/",
                        self.general_args.s3_bucket,
                    )
                    attempts = 3

                except Exception as e:
                    print("While saving checkpoint:", e)
                    attempts += 1

            shutil.rmtree(str(local_path))

    def sanity_check(self) -> None:
        print("Running sanity check")

        print("Checking validation")
        try:
            self.validation()
            print("Validation ok")
        except Exception as e:
            raise Exception(f"Validation fails with error: {e}")

        print("Checking saving")
        try:
            self._save_ckpt()
            print("Saving checkpoint ok")
        except Exception as e:
            raise Exception(f"Saving checkpoint fails with error: {e}")

        print("Checking training step")
        try:
            batch = next(iter(self.train_dataloader))
            self.training_step(batch, 0)
            print("Training ok")
        except Exception as e:
            raise Exception(f"Train step fails with error: {e}")

        self.sanity_check_complete = True

        print(
            "Sanity check complete, but remember -- real insanity is doing the same thing"
            " over and over and expecting different results"
        )


# if batch_idx == 3:
#     log_batch = log_processed_batch(
#         outputs["logits"],
#         outputs["loss_tensor"],
#         batch["input_ids"],
#         self.tokenizer,
#     )
# def log_processed_batch(logits, losses, batch, tokenizer):
#     all_examples = []
#     for example_id, example_logits in enumerate(logits):
#         response = process_example(
#             example_logits, losses[example_id], batch[example_id], tokenizer
#         )
#         all_examples.append(response)
#
#     html_log = "<br>----------------------<br>".join(all_examples)
#     to_log = {"Processed_val_batch": wandb.Html("".join(html_log))}
#
#     return to_log
