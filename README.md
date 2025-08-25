# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

PyTorch implementation of the paper [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135). This paper presents **FlashAttention**, an IO-aware exact attention algorithm that restructures attention to minimize slow GPU memory (HBM/DRAM) accesses by tiling Q, K, V into blocks, performing attention inside fast on-chip SRAM, and recomputing small pieces when needed. The result is an attention implementation that is both faster in wall-clock time and more memory-efficient (linear in sequence length), without approximating the attention math.

## FlashAttention’s core ideas:
- **IO-awareness & tiling:** split Q/K/V into tiles that fit in SRAM and stream them through nested loops so intermediate N×N attention matrices are never materialized in slow memory. This reduces HBM reads/writes compared to standard attention.
- **Recomputation for backward pass:** store compact normalization factors during the forward pass and recompute necessary quantities on-chip for the backward pass, avoiding storing the full attention matrix.
- **Kernel fusion & CUDA implementation:** fuse matmul → mask → stable softmax → dropout → matmul into a single CUDA kernel per tile to eliminate extra kernel launches and memory traffic; the authors provide a CUDA implementation and open-source code.

## Key theoretical claims:
- **IO complexity and optimality:** the paper analyzes HBM IO and shows FlashAttention requires asymptotically fewer HBM accesses O(N² d² M⁻¹) and gives a lower bound demonstrating no exact attention algorithm can asymptotically beat their HBM access count across SRAM sizes. 
- **Wall-clock speedups & quality:** FlashAttention yields large speedups in practice (examples include ~3× on GPT-2 for N=2K, 15% end-to-end training speedup on BERT-large, and multi× improvements on long-sequence benchmarks), enabling longer contexts that improve model quality (e.g., better perplexity and new capabilities on Path-X).

# Figure Analysis
![figure1](assets/figure1.png)
- FlashAttention idea: perform attention in tiles inside fast on-chip memory and fuse the attention pipeline to minimize memory traffic and kernel launches.

## 1) Memory hierarchy (left)
- GPUs have a small, very high-bandwidth on-chip SRAM (shared memory / L1) and a larger, slower HBM; CPU DRAM is even larger but far lower bandwidth.  
- FlashAttention minimizes transfers between HBM/DRAM and SRAM by performing as much work as possible inside SRAM, which is the key to reducing end-to-end latency.

---

## 2) Tiled attention algorithm (center)
1. **Tile Q, K, V** — split queries (Q), keys (K), and values (V) into tiles sized to fit SRAM/shared memory.  
2. **Copy Kᵀ tile to SRAM** — load a Kᵀ tile (or its transpose) into on-chip SRAM to reuse it across multiple Q tiles.  
3. **Copy Q tile to SRAM** — load one Q tile into SRAM (inner loop) and stream across Kᵀ tiles (outer loop).  
4. **Compute attention block in SRAM** — compute the small matrix multiply Q·Kᵀ for the current tile, apply masking, compute stable softmax, optionally apply dropout, and multiply by V — *all while keeping intermediate tensors in SRAM*.  
5. **Accumulate and write outputs** — accumulate partial outputs for the Q tile and write the final `sm(QKᵀ)V` result back to HBM/DRAM when complete.  
- The nested loops (outer over K/V tiles, inner over Q tiles) maximize data reuse and drastically reduce reads/writes to slow memory.

---

## 3) Performance implication (right)
- The PyTorch/naïve implementation performs many separate kernels (matmul, mask, softmax, dropout, another matmul), causing multiple HBM ↔ SRAM round trips per attention.  
- FlashAttention fuses those operations into a single compute kernel per tile (matmul + mask + softmax + dropout + matmul), which reduces memory traffic and yields a large wall-clock speedup (shown as a small single bar for the fused kernel).

# Performance Analysis
![figure2](assets/figure2.png)
- This figure illustrates how FlashAttention achieves efficiency gains by reducing memory traffic, tuning block size, and exploiting sparsity, compared against standard attention implementations.

## 1) Comparison with Standard Attention (left)
The table compares baseline attention with FlashAttention on three key metrics. Standard attention performs 66.6 GFLOPs, while FlashAttention executes slightly more operations (75.2 GFLOPs) due to recomputation, yet it drastically reduces HBM reads/writes from **40.3 GB down to 4.4 GB**. This sharp drop in off-chip memory traffic leads to large runtime improvements: the wall-clock time for FlashAttention is **7.3 ms**, nearly 6× faster than the 41.7 ms required for standard attention.

---

## 2) Effect of Block Size (center)
FlashAttention’s IO savings depend on the chosen block size used for tiling. Smaller blocks cause more HBM transfers, but as the block size increases, HBM accesses fall rapidly and then flatten once the block fully utilizes SRAM. Correspondingly, runtime decreases and stabilizes beyond a block size of 256. This demonstrates that FlashAttention’s efficiency is strongly tied to block size tuning, where larger tiles maximize reuse in SRAM without exceeding on-chip limits.

---

## 3) Sparsity Speedup (right)
FlashAttention can also leverage structured sparsity for further acceleration. The figure shows forward+backward runtime as a function of non-zero blocks. Dense FlashAttention defines the upper bound runtime, while block-sparse FlashAttention yields substantial speedups as sparsity increases, scaling nearly linearly with the fraction of blocks pruned. This highlights that FlashAttention integrates naturally with block-sparse attention methods, combining IO-awareness with sparsity for even greater efficiency.

---

## 4) Summary
These results confirm that FlashAttention’s speedup does not come from reducing FLOPs but from **minimizing memory traffic**, effectively rebalancing the bottleneck from being IO-bound to compute-bound. With appropriate block sizing and optional block-sparsity, FlashAttention achieves substantial runtime reductions while retaining exact attention computation.

# Runtime and Memory Usage
![figure3](assets/figure3.png)
- This figure compares FlashAttention with other attention implementations in terms of runtime (forward + backward) and memory footprint as sequence length increases.

## 1) Runtime Scaling (left)
The runtime graph shows how FlashAttention, block-sparse FlashAttention, PyTorch attention, Megatron attention, Linformer, and OpenAI Sparse Attention scale with sequence length. For short sequences, dense implementations such as PyTorch or Megatron attention appear competitive, but as sequence length grows the gap widens dramatically. FlashAttention exhibits much slower growth in runtime due to its IO-aware tiling strategy, while block-sparse FlashAttention improves further by skipping computations on pruned blocks. The plot highlights **crossover points** where FlashAttention becomes faster than other methods, especially at sequence lengths beyond 1K tokens, showing its practical advantage for long-context training and inference.

---

## 2) Memory Footprint (right)
The memory usage graph demonstrates the efficiency gains of FlashAttention. Standard PyTorch attention rapidly grows in memory cost, consuming **~20× more memory** than FlashAttention at moderate sequence lengths. Linformer attention grows more linearly but still uses about **2× more memory** than FlashAttention at 64K tokens. OpenAI Sparse Attention reduces some overhead but remains higher than FlashAttention at scale. By keeping all intermediate computations in SRAM and avoiding materialization of the full N×N attention matrix, FlashAttention achieves memory usage that scales nearly linearly with sequence length, enabling practical training at tens of thousands of tokens.

---

## 3) Summary
These results emphasize that FlashAttention outperforms traditional dense and approximate methods at longer sequence lengths. Its IO-awareness reduces runtime growth, while its memory efficiency enables contexts that would otherwise be infeasible on modern GPUs. Block-sparse FlashAttention extends these benefits further by combining exact tiling with sparsity, achieving even lower runtime without increasing memory footprint.

# FlashAttention vs. Block-Sparse FlashAttention

## FlashAttention
FlashAttention is an **IO-aware exact attention algorithm**. It restructures attention into tiled computations that fit inside fast on-chip SRAM, fuses all operations (matmul, masking, softmax, dropout, and value matmul) into a single kernel, and only writes the final outputs back to GPU HBM. This eliminates the need to materialize the full N×N attention matrix, reduces HBM reads/writes by an order of magnitude, and achieves substantial speedups without approximating attention. Importantly, FlashAttention is still **dense** attention: it computes over all query–key pairs exactly.

## Block-Sparse FlashAttention
Block-sparse FlashAttention extends this idea by combining IO-awareness with **structured sparsity**. Instead of computing over every Q–K interaction, the attention matrix is partitioned into fixed-size blocks, and only a subset of blocks are retained as non-zero. The FlashAttention tiling and fused kernel design naturally align with this block structure, so the algorithm skips entire tiles when they are known to be zero. This reduces both computation and memory traffic proportionally to the sparsity pattern, while still benefiting from the efficient on-chip execution of FlashAttention.

## Key Differences
- **Computation:**  
  - FlashAttention computes dense attention exactly over all tokens.  
  - Block-sparse FlashAttention only computes a subset of attention blocks, reducing FLOPs and runtime.  

- **Memory:**  
  - FlashAttention reduces memory overhead to linear in sequence length but still processes all pairs.  
  - Block-sparse FlashAttention further reduces memory by skipping storage of zeroed-out tiles.  

- **Use case:**  
  - FlashAttention is optimal when dense attention is required for full accuracy.  
  - Block-sparse FlashAttention is useful when tasks or architectures allow structured sparsity (e.g., long-sequence transformers, local + global attention patterns), trading some generality for further efficiency.  

## Summary
FlashAttention and block-sparse FlashAttention share the same IO-aware, fused kernel foundation, but block-sparse FlashAttention layers sparsity on top of that design. This makes it even faster and more memory-efficient for very long sequences, while FlashAttention remains the drop-in exact dense attention replacement.

## Configuration
```python
class GPT3Config:
    def __init__(self):
        # Model architecture
        self.vocab_size = 50257     # Size of the GPT-3 tokenizer vocabulary
        self.d_model = 768          # Model hidden dimension (GPT-3 175B uses 12288)
        self.n_layers = 12          # Number of Transformer decoder layers (GPT-3 175B uses 96 layers)
        self.n_heads = 12           # Number of attention heads per layer (GPT-3 175B uses 96 heads)
        self.d_ff = 3072            # Feed-forward network hidden dimension (GPT-3 175B uses 49152)
        self.dropout = 0.1          # Dropout rate (GPT-3 paper did not use dropout)
        self.max_seq_len = 512      # Maximum sequence length (GPT-3 uses up to 2048 tokens)

        # Optimization / hyperparameters
        self.lr = 1e-4              # Learning rate for Adam optimizer
        self.betas = (0.9, 0.95)    # Beta values for Adam optimizer
        self.eps = 1e-8             # Epsilon for numerical stability in Adam optimizer
        self.weight_decay = 0.0     # Weight decay for regularization
        self.warmup_steps = 1000    # Number of steps to linearly warm up the LR
        self.lr_decay = "cosine"    # Learning rate decay schedule after warmup

        # FlashAttention / blocking
        self.block_size = 128             # Block size used by block-sparse
        self.use_flash_attention = True   # Flag to enable FlashAttention

        # Model details
        self.activation = "gelu"          # Activation function used in feed-forward layers
        self.initializer_range = 0.02     # Stddev for weight initialization

        # A100 optimization setup
        self.gradient_accumulation_steps = 16  # Large batch simulation
        self.mixed_precision = True            # FP16/BF16
        self.compile_model = True              # torch.compile
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Automatically select GPU if available, else use CPU
        self.epochs = 5                        # Number of epochs to train
```

=========================================================================================
=========================================================================================
## Experiment Note

I was running the experiments on **Google Colab with an A100 GPU**,  
but due to **limited computing units**, I was unable to provide the results at this time.  

I plan to **re-run the experiments in the future** when units are available,  
and will update this repository with the **verification results** accordingly.  
=========================================================================================
=========================================================================================

# Additional Insights

## 1) IO Complexity Lower Bound & Optimality
The paper doesn't just present FlashAttention’s improved I/O characteristics—it **proves** that under realistic GPU SRAM size assumptions, **no exact attention algorithm** can asymptotically reduce off-chip memory accesses (HBM reads/writes) beyond FlashAttention’s strategy. This lower bound establishes that FlashAttention is not just empirically efficient but **optimally memory-efficient** within the exact-attention regime.

## 2) Memory-Efficient Backward Pass via Recomputation
FlashAttention avoids storing the entire softmax-backed intermediate attention matrix by saving compact normalization factors (like per-block max and sum) and RNG state (for dropout). These enable recomputing necessary values **on-chip** during the backward pass, trading additional compute for drastically reduced HBM storage and access. The backward pass thus remains memory-efficient while staying exact.

## 3) Algorithmic Details: "Online" Softmax Decomposition
To support tiling and streaming, FlashAttention applies the softmax in a blockwise, incremental (“online”) fashion. Each tile computes local statistics (max and sum of exp) which are combined across tiles to achieve correct softmax with numerical stability—without ever loading the full QKᵀ or attention matrix into memory.

## 4) Extensions: Block-Sparse FlashAttention
The paper formalizes **block-sparse FlashAttention**, where a pre-defined block-sparsity is used to skip zero tiles entirely. This version offers IO complexity reduced by a factor proportional to the fraction of nonzero blocks, making it even faster and more memory-efficient when sparsity applies.
