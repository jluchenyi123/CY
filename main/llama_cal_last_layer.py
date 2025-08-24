# run_last_layer_contrib.py
import argparse
import json
import torch
from model_.llama import LlamaAttributor, AttrConfig
from datasets import load_dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Compute last-layer token contributions using LlamaAttributor.")
    parser.add_argument("--model_path", type=str, default="/home/YiChen/uq/model/Llama-3.2-3B-Instruct", help="HF 模型路径或名称")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_path", type=str, default="/home/YiChen/uq/data/train-v2.0-flatten.json")
    parser.add_argument("--max_samples", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--metric", type=str, default="manhattan", choices=["manhattan", "dot"])
    parser.add_argument("--i_block", type=int, default=128, help="分块大小，显存不够可调小")
    parser.add_argument("--save_json", type=str, default="/home/YiChen/uq/dev_fig/fig_statistic/manhattan_2.json", help="可选：保存结果到该 JSON 路径")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    dataset = load_dataset("json", data_files={"validation": args.data_path})["validation"]
    if args.max_samples is not None and args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    cfg = AttrConfig(
    model_path=args.model_path,
    device=args.device,
    dtype=args.dtype,
    metric=args.metric,
    i_block=args.i_block,
)
    tool = LlamaAttributor(cfg)
    results = []

    for example in tqdm(dataset, total=len(dataset)):
        context = example["context"]
        question = example["question"]
        print(f"Context: {context}\nQuestion: {question}\n")
        prompt = tool.build_prompt(context, question)
        enc = tool.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(args.device) for k, v in enc.items()}
        input_ids = enc["input_ids"]  # [1, T]
        batch_size, seq_len = input_ids.shape
        tokens = [tool.tokenizer.decode([tid], skip_special_tokens=False)
                for tid in input_ids[0].tolist()]
        oproj_refs = tool._capture_o_proj_refs(enc)
        attn_res_blocks, attn_after_blocks = tool._manual_forward_once(input_ids, oproj_refs)
        contrib_mats = tool._contrib_matrices(attn_res_blocks, attn_after_blocks)
        result = tool._accumulate_last_token(contrib_mats, seq_len)
        token_contrib_pairs = [tokens, result]
        results.append(token_contrib_pairs)
        print(f"{token_contrib_pairs}\n")
    # 可选保存
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to: {args.save_json}")

if __name__ == "__main__":
    main()


