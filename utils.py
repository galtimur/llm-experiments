import torch
import copy
import wandb


def filter_grad(model, mask_dict, threshold, type, apply_saved_mask):
    num_el = 0
    num_grad = 0
    num_common = 0
    for name, param in model.named_parameters():
        # grad_norm2 = torch.norm(param.grad.data)
        if param.grad.data.ndim < 2:
            continue
        num_el += param.grad.data.numel()
        if not apply_saved_mask:
            param_abs = torch.abs(param.grad.data)
            grad_norm = torch.mean(param_abs)
            grad_max = torch.max(param_abs)
            grad_min = torch.min(param_abs)
            if type == "largest":
                thresh = grad_norm / threshold
                if thresh > grad_max:
                    thresh = (1 - threshold) * grad_max
                mask = param_abs > thresh
            if type == "smallest":
                thresh = grad_norm * threshold
                if thresh < grad_min:
                    thresh = (1 + threshold) * grad_min
                mask = param_abs < thresh
        else:
            mask = mask_dict[name]
        if torch.sum(mask) > 0:
            param.grad.data = param.grad.data * mask.float()
            # param.grad.data = (
            #     param.grad.data
            #     / (torch.norm(param.grad.data) + 0.1 * grad_norm2)
            #     * grad_norm2
            # )
            if name in mask_dict:
                common_mask = mask_dict[name] & mask
                num_common += torch.sum(common_mask)
            num_grad += torch.sum(mask)
            mask_dict[name] = mask

    return num_grad / num_el, num_common / num_grad, mask_dict


def track_params(model, model_start, model_previous, model_start_1):
    num_el = dist = dist1 = grad_step = dev_step = sum_step = sum_dist = dev_dist = 0
    for (name, param), (_, param_start), (_, param_prev), (_, param_start_1) in zip(
        model.named_parameters(),
        model_start.named_parameters(),
        model_previous.named_parameters(),
        model_start_1.named_parameters(),
    ):
        dist_tensor = param.data - param_start.data
        dist1_tensor = param.data - param_start_1.data
        step = param.data - param_prev.data

        dist += torch.norm(dist_tensor) ** 2
        dist1 += torch.norm(dist1_tensor) ** 2
        grad_step += torch.norm(step) ** 2
        num_el += param.data.numel()

        sum_step += torch.sum(step)
        dev_step += torch.sum((step - sum_step / num_el) ** 2)

        sum_dist += torch.sum(dist_tensor)
        dev_dist += torch.sum((dist_tensor - sum_dist / num_el) ** 2)
        # mean_dist += torch.sum(dist_tensor)
        # param_abs = torch.abs(param.grad.data)
        # grad_norm = torch.mean(param_abs)
        # grad_max = torch.max(param_abs)
        return (
            torch.sqrt(dist / num_el),
            torch.sqrt(dist1 / num_el),
            torch.sqrt(grad_step / num_el),
            torch.sqrt(dev_step / num_el),
            torch.sqrt(dev_dist / num_el),
        )


def process_batch_template(batch, tokenizer, max_seq_length):
    texts = [item["text"] for item in batch]
    inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt",
    )

    input_ids = inputs.input_ids
    labels = input_ids.clone().contiguous()
    labels[labels == tokenizer.pad_token_id] = -100
    attn_mask = labels != -100
    inputs["labels"] = labels  # [:, 1:]
    inputs["input_ids"] = inputs.input_ids.contiguous()  # [:, :-1]
    inputs["labels"][inputs["input_ids"] == tokenizer.pad_token_id] = -100
    inputs["attn_mask"] = attn_mask

    return inputs


def validate(val_loader, model, device):
    total_eval_loss = 0
    model.eval()
    for n, batch in enumerate(val_loader, start=1):
        with torch.no_grad():
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn_mask = batch["attn_mask"].to(device)

            outputs = model(input_ids=inputs, labels=labels, attention_mask=attn_mask)
            eval_loss = outputs[0]
            total_eval_loss += eval_loss.item()
    val_loss = total_eval_loss / n
    print(f"Validation loss = {val_loss:.2f}")
    return val_loss


def general_train_step(
    batch,
    model,
    model_start,
    model_start1,
    mask_dict,
    optimizer,
    batch_accum,
    consumed_batches,
    epoch,
    args,
    device,
    filter_gradients,
    to_log,
):
    inputs = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attn_mask = batch["attn_mask"].to(device)
    batch_size = inputs.shape[0]
    consumed_samples = batch_size * consumed_batches

    loss = model(input_ids=inputs, labels=labels, attention_mask=attn_mask)[0]
    loss.backward()

    if consumed_samples % batch_accum == 0:
        log_dict = {}
        model_prev = copy.deepcopy(model)
        if filter_gradients:
            grad_part, part_common, mask_dict = filter_grad(
                model,
                mask_dict,
                args["threshold"],
                args["type"],
                apply_saved_mask=False,
            )
            log_dict = {"grad part": grad_part, "part common": part_common}
        else:
            log_dict = {}
        optimizer.step()
        if args["optimizer"] == "AdamW":
            # adjust_model_mask(model, model_prev, mask_dict)
            pass
        dist, dist1, grad_step, std_step, std_dist = track_params(
            model, model_start, model_prev, model_start1
        )
        optimizer.zero_grad()
        if epoch == 0:
            dist1 = 0
        log_dict.update(
            {
                "distance": dist,
                "distance from 1 epoch": dist1,
                "grad step": grad_step,
                "std distance": std_dist,
                "std step": std_step,
            }
        )
        if to_log:
            log_dict.update(
                {"loss vs samples": loss.item(), "samples": consumed_samples}
            )
            wandb.log(log_dict, commit=True)
    return mask_dict
