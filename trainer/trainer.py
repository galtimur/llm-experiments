import random
import shutil
import warnings
from math import ceil
from pathlib import Path

import torch
import torch.nn.functional as tofu
import wandb
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)

from data.s3_data_exchange import upload_directory_s3


class Trainer:
    def __init__(
        self,
        config,
        train_data_loader,
        val_data_loader,
        perform_sanity_check=True,
    ):
        self.train_dataloader = train_data_loader
        self.val_dataloader = val_data_loader
        self.sanity_check_complete = False

        self.config = config
        self.train_args = config.train
        self.device = self.config.device
        random_id = random.randint(1, 100000)
        self.wand_run_name = f"{self.config.model[:6]}_{self.config.parent_dataset}_len-{self.config.sequence_length}_batch-{self.config.accumulate_grad}_lr-{self.config.max_lr}--id{random_id}"

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.name)

        self.scheduler_mapping = {
            "linear": get_linear_schedule_with_warmup,
            "cosine": get_cosine_schedule_with_warmup,
        }

        self.model.to(self.device)
        self.accum_steps = config.train.accumulate_grad

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.train.max_lr,
            weight_decay=self.config.train.weight_decay,
        )

        self.scheduler = None
        self.define_scheduler()

        self.checkpoint_steps = []
        self.val_steps = []
        self.setup_validation_and_checkpoint_schedule()

        self.batches_done = 0
        self.mini_batches_done = 0
        self.loss_acc = 0

        if perform_sanity_check:
            self.sanity_check()

    def _compute_steps(self):
        # steps|samples counters
        # Prioritization: samples-steps-ratio
        # samples = steps*batch
        step_args = self.train_args.steps
        full_batch_size = self.train_args.train_batch_size

        if step_args.warmup_samples is not None:
            step_args.warmup_steps = step_args.warmup_samples // full_batch_size
        if step_args.val_every_sample is not None:
            step_args.val_every_step = step_args.val_every_sample // full_batch_size
        if step_args.save_every_sample is not None:
            step_args.save_every_step = step_args.save_every_sample // full_batch_size
        if step_args.max_train_samples is not None:
            step_args.max_train_steps = step_args.max_train_samples // full_batch_size
        if step_args.max_val_samples is not None:
            step_args.max_val_steps = (
                step_args.max_val_samples // self.train_args.val_batch_size
            )

    def _compute_warmup(self):
        if type(self.config["warmup_steps"]) is int:
            return self.config["warmup_steps"]
        elif type(self.config["warmup_steps"]) is float:
            total_steps = (
                len(self.train_dataloader) // self.accum_steps
            ) * self.config["num_epochs"]
            return ceil(self.config["warmup_steps"] * total_steps)
        else:
            raise ValueError(
                f"Invalid value for `warmup_steps`, got {type(self.config['warmup_steps'])}"
            )

    def define_scheduler(self):
        warmup_steps = self._compute_warmup()
        scheduler_fetcher = self.scheduler_mapping[self.config.train.scheduler]
        scheduler_fetcher(
            self.opt,
            warmup_steps,
            (len(self.train_dataloader) // self.accum_steps)
            * self.config.train.num_epochs,
        )

    def run_epoch(self):
        if not self.sanity_check_complete:
            self.validation()
            self.save_hf_checkpoint()
            self.sanity_check_complete = True

        # TODO replace or add samples
        wandb.log({"train/total_minibatches": len(self.train_dataloader)})

        for batch_idx, batch in tqdm(
            enumerate(self.train_dataloader), total=len(self.train_dataloader)
        ):
            to_log = self.training_step(batch, batch_idx)
            wandb.log(to_log)

            if len(self.train_dataloader) < self.accum_steps:
                warnings.warn(
                    "No optimizer steps will be done since self.accum_steps > num_of_batches_in_dataloader"
                )

            if self.mini_batches_done in self.val_steps:
                print(f"validation on step {self.batches_done}")
                to_log = self.validation()
                wandb.log(to_log)
                self.val_steps.remove(self.mini_batches_done)

            if self.mini_batches_done in self.checkpoint_steps:
                print(f"checkpoint on step: {self.batches_done}")
                self.save_hf_checkpoint()
                self.checkpoint_steps.remove(self.mini_batches_done)

        to_log = self.validation()
        wandb.log(to_log)
        self.save_hf_checkpoint()

    def run_training(self):
        total_epochs = self.config["num_epochs"]
        for epoch_id in range(total_epochs):
            print(f"Epoch {epoch_id} / {total_epochs} start")
            self.run_epoch()

    def validation(self):
        self.model.eval()

        log_humaneval = self.generate_humaneval_task()

        full_val_loss = 0
        for batch_idx, batch in enumerate(self.val_dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model_step(batch, return_dict=True)
                full_val_loss += outputs["loss"]

            if batch_idx == 3:
                log_batch = log_processed_batch(
                    outputs["logits"],
                    outputs["loss_tensor"],
                    batch["input_ids"],
                    self.tokenizer,
                )
        val_loss_mean = full_val_loss / len(self.val_dataloader)

        # TODO: refactor duplicated code?
        full_humaneval_loss = 0
        for batch_idx, batch in enumerate(self.human_eval_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model_step(batch, return_dict=True)
                full_humaneval_loss += outputs["loss"]

        humaneval_loss_mean = full_humaneval_loss / len(self.val_dataloader)

        self.model.train()

        to_log = {
            "val/val_loss_mean": val_loss_mean,
            "val/humaneval_loss_mean": humaneval_loss_mean,
        }
        to_log.update(log_batch)
        to_log.update(log_humaneval)

        return to_log

    def model_step(self, batch, return_dict=False):
        outputs = self.model(**batch, return_dict=True)
        targets = batch["input_ids"][:, 1:]
        logits = outputs["logits"][:, :-1]
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

    def training_step(self, batch, batch_idx):
        to_log = {}

        batch = {k: v.to(self.device) for k, v in batch.items()}

        loss = self.model_step(batch) / self.accum_steps
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_norm, norm_type=2.0
        )

        to_log["train/loss_mini_batch"] = loss
        to_log["grad_norm"] = grad_norm

        self.loss_acc += loss.item()
        if (batch_idx % self.accum_steps) == (self.accum_steps - 1):
            self.opt.step()
            self.opt.zero_grad()
            lr = next(iter(self.opt.param_groups))["lr"]
            self.scheduler.step()

            to_log["train/loss_batch"] = self.loss_acc
            to_log["lr"] = lr

            self.batches_done += 1
            self.loss_acc = 0.0

        self.mini_batches_done += 1
        return to_log

    def save_ckpt(self):
        pth = self.config.get("model_checkpoints")
        local_path = Path(f"{pth}/{self.wand_run_name}-{self.batches_done}")
        local_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(local_path))
        self.tokenizer.save_pretrained(str(local_path))

        if self.config.upload_to_s3:
            s3_path = self.config.get("s3_checkpoint_path")
            attempts = 0
            while attempts < 3:
                try:
                    upload_directory_s3(
                        local_path,
                        f"{s3_path}/{self.wand_run_name}-{self.batches_done}/",
                        self.config.get("s3_bucket"),
                    )
                    attempts = 3

                except Exception as e:
                    print("While saving checkpoint:", e)
                    attempts += 1

            shutil.rmtree(str(local_path))

    def sanity_check(self):
        print("Running sanity check")

        print("Checking validation")
        try:
            self.validation()
            print("Validation ok")
        except Exception as e:
            raise Exception(f"Validation fails with error: {e}")

        print("Checking saving")
        try:
            self.save_hf_checkpoint()
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

    def setup_validation_and_checkpoint_schedule(self):
        self.checkpoint_steps = list(
            range(
                1,
                len(self.train_dataloader) * self.config["num_epochs"],
                self.config.save_hf_checkpoints_every * self.accum_steps,
            )
        )
        self.val_steps = list(
            range(
                1,
                len(self.train_dataloader) * self.config["num_epochs"],
                self.config.validate_every * self.accum_steps,
            )
        )

        print(
            f"Validation every: {self.config.validate_every * self.accum_steps}, in total: {len(self.val_steps)} validations"
        )
        print(
            f"Saving checkpoints every: {self.config.save_hf_checkpoints_every * self.accum_steps}, in total: {len(self.checkpoint_steps)} saves"
        )


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
