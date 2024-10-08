import gzip
import os
import shutil
from math import ceil
from pathlib import Path

import torch
import torch.nn.functional as tofu
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (  # StoppingCriteria,; StoppingCriteriaList,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

import wandb
from data.s3_data_exchange import upload_directory_s3


def compress_model(input_file_path, output_file_path):
    with open(input_file_path, "rb") as f_in:
        with gzip.open(output_file_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# TODO add wandb_init
class Trainer:
    def __init__(
        self,
        config,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        perform_sanity_check: bool = True,
        model=None,
        tokenizer=None,
    ):
        self.train_dataloader = train_data_loader
        self.val_dataloader = val_data_loader

        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_args = config.train
        self.model_args = config.model
        self.general_args = self.config.general
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sanity_check_complete = self.train_args.do_sanity_check
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
        self.processed_tokens = 0

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
        if (self.model is not None) and (self.tokenizer is not None):
            model = self.model
            tokenizer = self.tokenizer
        elif self.model_args.pretrained:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_args.name,
                torch_dtype=dtype,
                attn_implementation=flash_attn,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_args.name)
        else:
            config_model = AutoConfig.from_pretrained(self.model_args.name)
            # model_pars = self.model_args.parameters
            # config_model.hidden_size = model_pars["hidden_dim"]
            # config_model.intermediate_size = model_pars["ff_dim"]
            # config_model.num_hidden_layers = model_pars["num_layers"]
            model = AutoModelForCausalLM.from_config(
                config_model, torch_dtype=dtype, attn_implementation=flash_attn
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_args.name)
        num_pars = count_parameters(model)
        print(f"Number of model parameters = {num_pars/1e6:.0f}M")
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
        max_train_steps = (
            step_args.num_epochs
            * len(self.train_dataloader)
            * self.train_args.train_mini_batch_size
            // full_batch_size
        )
        if step_args.max_train_samples is not None:
            step_args.max_train_steps = min(
                step_args.max_train_samples // full_batch_size, max_train_steps
            )
        elif step_args.max_train_steps is None:
            step_args.max_train_steps = max_train_steps
        else:
            step_args.max_train_steps = min(
                step_args.max_train_steps, step_args.max_train_steps
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

    def define_scheduler(self) -> None:
        scheduler_fetcher = self.scheduler_mapping[self.train_args.scheduler]
        return scheduler_fetcher(
            optimizer=self.opt,
            num_warmup_steps=self.step_args.warmup_steps,
            num_training_steps=self.step_args.max_train_steps,
        )

    def wandb_init(self):

        # random_id = random.randint(1, 100000)
        model_name = self.config.model.name.split("/")[-1]
        os.environ["WANDB_BASE_URL"] = self.config.general.wandb_url
        self.wandb_run_name = model_name
        # self.wand_run_name = f"{self.model_args.name[:6]}_len-{self.model_args.sequence_length}_bs-{self.train_args.train_batch_size}_lr-{self.train_args.max_lr}--id{random_id}"
        # wandb_run_name += f"_cr_{self.train_args.compression_rate}"
        # wandb_run_name += f"_seg_{self.train_args.segment_length}"
        # wandb_run_name += f"_batch_{self.batch_size_global}"
        # wandb_run_name += f"{model_name}"
        wandb.init(
            project=self.config.general.wandb_project,
            entity=self.config.general.wandb_entity,
            config=self.config,
            name=self.wandb_run_name,
        )
        wandb.define_metric("tokens")
        wandb.define_metric("train/loss vs tokens", step_metric="tokens")
        wandb.define_metric("val/loss vs tokens", step_metric="tokens")
        wandb.define_metric("train/compress_ratio vs tokens", step_metric="tokens")

        wandb.run.log_code(".")

    def run_epoch(self) -> None:
        pbar = tqdm(total=self.step_args.max_train_steps)
        for batch_idx, batch in enumerate(self.train_dataloader, start=1):
            self.processed_tokens += batch["input_ids"].numel()
            to_log = self.training_step(batch, batch_idx)
            if to_log is not None:
                to_log.update({"tokens": self.processed_tokens})
                wandb.log(to_log)
                pbar.update()

            if (to_log is not None) and (
                (self.batches_done - 1) % self.step_args.val_every_step == 0
            ):
                print(f"validation on step {self.batches_done}")
                to_log = self.validation()
                to_log.update({"tokens": self.processed_tokens})
                wandb.log(to_log)

            if (to_log is not None) and (
                (self.batches_done - 1) % self.step_args.save_every_step == 0
            ):
                print(f"checkpoint on step: {self.batches_done}")
                model_path = self._save_ckpt()
                compress_ratio = self.calculate_zip_ratio(model_path)
                log_dict = {
                    "tokens": self.processed_tokens,
                    "train/compress_ratio": compress_ratio,
                    "train/compress_ratio vs tokens": compress_ratio,
                }
                wandb.log(log_dict)

            if self.batches_done > self.step_args.max_train_steps:
                break

        to_log = self.validation()
        wandb.log(to_log)
        self._save_ckpt()

    def run_training(self) -> None:
        self.wandb_init()
        if not self.sanity_check_complete:
            self.validation(limit=20)
            self._save_ckpt()
            self.sanity_check_complete = True
        total_epochs = self.step_args.num_epochs
        for epoch_id in range(total_epochs):
            print(f"Epoch {epoch_id} / {total_epochs} start")
            self.run_epoch()

    def validation(self, limit: int = -1) -> dict[str, float]:
        self.model.eval()
        full_val_loss = 0
        if limit < 0:
            limit = len(self.val_dataloader)
        for batch_idx, batch in tqdm(
            enumerate(self.val_dataloader, start=1), total=limit
        ):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad(), torch.autocast(device_type="cuda"):
                outputs = self.model_step(batch, return_dict=True)
                full_val_loss += outputs["loss"].item()

            if batch_idx > limit:
                break

        val_loss_mean = full_val_loss / batch_idx
        to_log = {
            "val/loss": val_loss_mean,
            "val/loss vs tokens": val_loss_mean,
        }
        self.model.train()
        # to_log.update(log_batch)

        return to_log

    def model_step(self, batch, return_dict: bool = False):
        outputs = self.model(**batch, return_dict=True)
        targets = batch["input_ids"][:, 1:]
        logits = outputs["logits"][:, :-1]
        # Is it neccesary?
        # logits = logits.to(torch.float32)
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

        with torch.autocast(device_type="cuda"):
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
            to_log["train/loss vs tokens"] = self.loss_acc

            to_log["lr"] = lr

            self.batches_done += 1
            self.loss_acc = 0.0
            to_log["grad_norm"] = grad_norm
            to_log["samples"] = batch_idx * self.train_args.train_mini_batch_size
            return to_log

        self.mini_batches_done += 1
        return None

    @staticmethod
    def calculate_zip_ratio(model_folder):
        model_file = model_folder / "model.safetensors"
        compressed_file = model_folder.parent / f"{model_folder.name}_model.zip"
        compress_model(model_file, compressed_file)
        model_size = os.path.getsize(model_file)
        gzip_size = os.path.getsize(compressed_file)
        compression_ratio = gzip_size / model_size
        os.remove(compressed_file)

        return compression_ratio

    def _save_ckpt(self) -> None:
        pth = self.general_args.checkpoints_path
        local_path = Path(f"{pth}/{self.wandb_run_name}-{self.batches_done}")
        local_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(local_path))
        self.tokenizer.save_pretrained(str(local_path))
        if self.general_args.upload_to_s3:
            self.save_s3(local_path)

        return local_path

    def _save_s3(self, local_path):
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
