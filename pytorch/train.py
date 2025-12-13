#@title GPT-3 Training

# Enables cuDNN autotuner to find the best algorithm for the hardware (improves training speed)
torch.backends.cudnn.benchmark = True

# Initialize Transformer model
model = Transformer(
    vocab_size=config.vocab_size,
    d_model=config.d_model,
    num_heads=config.n_heads,
    num_layers=config.n_layers,
    d_ff=config.d_ff,
    dropout=config.dropout,
    max_len=max_seq_len
).to(config.device)

# Compile model if requested
if getattr(config, "compile_model", False):
    try:
        model = torch.compile(model, backend="inductor")
    except Exception as e:
        print("[WARN] torch.compile failed or incompatible:", e)

# Mixed precision scaler
scaler = torch.cuda.amp.GradScaler(enabled=bool(getattr(config, "mixed_precision", True)))

# Optimizer & Scheduler
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.AdamW(
    model.parameters(),
    lr=config.lr,
    betas=config.betas,
    eps=config.eps,
    weight_decay=config.weight_decay
)

# Gradient accumulation & steps
grad_accum_steps = max(1, getattr(config, "gradient_accumulation_steps", 1))
steps_per_epoch = len(train_loader) // grad_accum_steps
# Total number of optimization steps across all epochs
total_steps = steps_per_epoch * config.epochs if steps_per_epoch > 0 else len(train_loader) * config.epochs

# CosineAnnealingLR learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

# Stats
flash_stats = {'time_per_batch': [], 'peak_memory_mb': []}
log_interval = 20

# --- Training loop ---
model.train()
for epoch in range(config.epochs):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=config.device)

    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)   # Reset gradients efficiently

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch {epoch+1}/{config.epochs}", leave=True)

    for batch_idx, (input_ids, target_ids) in pbar:
        # move to GPU efficiently
        input_ids = input_ids.to(config.device, non_blocking=True)
        target_ids = target_ids.to(config.device, non_blocking=True)

        # timing
        if torch.cuda.is_available():
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
        else:
            start_time = time.time()

        # forward + mixed precision
        with torch.cuda.amp.autocast(enabled=config.mixed_precision):
            logits = model(input_ids) # Forward pass
            loss = criterion(logits.view(-1, config.vocab_size), target_ids.view(-1))
            loss = loss / grad_accum_steps  # Scale loss for gradient accumulation

        # backward with scaling
        scaler.scale(loss).backward()

        # optimizer step with gradient accumulation
        if ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == len(train_loader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(config, 'max_grad_norm', 1.0))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            try:
                scheduler.step()  # Update learning rate
            except Exception:
                pass

        # timing measurement
        if torch.cuda.is_available():
            end_evt.record()
            torch.cuda.synchronize()
            batch_time = start_evt.elapsed_time(end_evt) / 1000.0   # Convert ms to seconds
            peak_memory = torch.cuda.max_memory_allocated(device=config.device) / 1024**2   # MB
        else:
            batch_time = time.time() - start_time
            peak_memory = 0.0

        # Save stats
        flash_stats['time_per_batch'].append(batch_time)
        flash_stats['peak_memory_mb'].append(peak_memory)

        # Update loss tracking
        total_loss += loss.item() * grad_accum_steps
        avg_loss = total_loss / (batch_idx + 1)

        # log periodically
        if (batch_idx % log_interval == 0) or (batch_idx + 1 == len(train_loader)):
            try:
                current_lr = scheduler.get_last_lr()[0]
            except Exception:
                current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'lr': f"{current_lr:.3e}",
                'time(s)': f"{batch_time:.4f}",
                'peak_mem(MB)': f"{peak_memory:.1f}"
            })

    # epoch cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.save(model.state_dict(), f"gpt3_epoch{epoch+1}.pt")
    print(f"Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}, checkpoint saved.")

# --- final stats ---
if flash_stats['time_per_batch']:
    avg_time = sum(flash_stats['time_per_batch']) / len(flash_stats['time_per_batch'])
    avg_mem = sum(flash_stats['peak_memory_mb']) / len(flash_stats['peak_memory_mb'])
else:
    avg_time, avg_mem = 0.0, 0.0

print(f"--- Training Summary ---")
print(f"Average batch time: {avg_time:.6f} sec")
print(f"Average peak memory: {avg_mem:.2f} MB")
print("Training complete.")
