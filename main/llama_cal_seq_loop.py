# run_last_layer_contrib.py
import argparse
import json
import torch
from model_.llama import LlamaAttributor, AttrConfig
from datasets import load_dataset
from tqdm import tqdm

import re

STOP_PUNCTS = {".", "!", "?"}

def _extract_answer_span(text: str) -> str:
    # 取最后一个 " Answer:" 之后的内容
    key = " Answer:"
    pos = text.rfind(key)
    return "" if pos == -1 else text[pos + len(key):]

def _count_words(s: str) -> int:
    # 简单英文词计数（字母/数字/撇号）
    return len(re.findall(r"[A-Za-z0-9']+", s))

def generate_until_stop(tool, context: str, question: str,
                        word_limit: int = 5,
                        max_new_tokens: int = 20,
                        do_sample: bool = False,
                        temperature: float = 1.0,
                        top_p: float = 1.0):
    prompt = tool.build_prompt(context, question)
    generated = ""

    for _ in range(max_new_tokens):
        next_id, next_txt = tool.generate_one_token(
            prompt, do_sample=do_sample, temperature=temperature, top_p=top_p
        )

        # 1) eos / 特殊结束符
        if tool.tokenizer.eos_token_id is not None and next_id == tool.tokenizer.eos_token_id:
            break
        if next_txt.strip() in {"<|eot_id|>", "<|endoftext|>", "</s>"}:
            break

        # 先把 token 接上
        prompt += next_txt
        generated += next_txt

        # 2) 词数上限
        ans = _extract_answer_span(prompt)
        if _count_words(ans) >= word_limit:
            # 如果最后一个 token 还是字母/数字，通常再给一个标点更自然；
            # 你也可以直接 break，不再补标点
            break

        # 3) 换行或句末标点（且已有内容）
        if next_txt == "\n" and _count_words(ans) > 0:
            break
        if next_txt.strip() in STOP_PUNCTS and _count_words(ans) > 0:
            break

    return {
        "final_prompt": prompt,
        "generated_answer": _extract_answer_span(prompt).strip()
    }

def loop_result(tool, device: str, prompt: str, word_limit: int = 800, results = []):
    init_enc = tool.tokenizer(prompt, return_tensors="pt")
    init_enc = {k: v.to(device) for k, v in init_enc.items()}
    init_input_ids = init_enc["input_ids"]  # [1, T]
    init_tokens = [tool.tokenizer.decode([tid], skip_special_tokens=False)
                    for tid in init_input_ids[0].tolist()]
    batch_size, seq_len_init = init_input_ids.shape
    results.append(init_tokens)
    score_list = []
    for _ in range(word_limit):
        next_id, next_txt = tool.generate_one_token(
            prompt, do_sample=False, temperature=1.0, top_p=1.0)
        # 1) eos / 特殊结束符
        if tool.tokenizer.eos_token_id is not None and next_id == tool.tokenizer.eos_token_id:
            break
        if next_txt.strip() in {"<|eot_id|>", "<|endoftext|>", "</s>"}:
            break

        # 先把 token 接上
        prompt += next_txt
        # generated += next_txt

        enc = tool.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        input_ids = enc["input_ids"]  # [1, T]
        batch_size, seq_len = input_ids.shape
        tokens = [tool.tokenizer.decode([tid], skip_special_tokens=False)
                for tid in input_ids[0].tolist()]
        oproj_refs = tool._capture_o_proj_refs(enc)
        attn_res_blocks, attn_after_blocks = tool._manual_forward_once(input_ids, oproj_refs)
        # 算完分的贡献矩阵
        contrib_mats = tool._contrib_matrices(attn_res_blocks, attn_after_blocks)
        result = tool._accumulate_last_token(contrib_mats, seq_len)
        result_init_tokens_contrib = result[:seq_len_init]
        # token_contrib_pairs = [init_tokens, result_init_tokens]
        score_list.append(result_init_tokens_contrib)
    results.append(score_list)
    return prompt, results

def main():
    parser = argparse.ArgumentParser(description="Compute last-layer token contributions using LlamaAttributor.")
    parser.add_argument("--model_path", type=str, default="/home/YiChen/uq/model/Llama-3.2-3B-Instruct", help="HF 模型路径或名称")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--data_path", type=str, default="/home/YiChen/uq/data/train-v2.0-flatten.json")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--metric", type=str, default="manhattan", choices=["manhattan", "dot"])
    parser.add_argument("--i_block", type=int, default=128, help="分块大小，显存不够可调小")
    parser.add_argument("--save_json", type=str, default="/home/YiChen/uq/dev_fig/fig_statistic/full_loop.json", help="可选：保存结果到该 JSON 路径")
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

    # for example in tqdm(dataset, total=len(dataset)):
    #     context = example["context"]
    #     question = example["question"]
    #     print(f"Context: {context}\nQuestion: {question}\n")
    #     prompt = tool.build_prompt(context, question)

    #     results_loop = loop_result(tool=tool,device=cfg.device,prompt=prompt,word_limit=5,results=results)
    #     print(f"{results_loop}\n")
    # --------test--------
    example = dataset[999]
    context = example["context"]
    question = example["question"]
    print(f"Context: {context}\nQuestion: {question}\n")
    prompt = tool.build_prompt(context, question)

    Prompt, results_loop = loop_result(tool=tool,device=cfg.device,prompt=prompt,word_limit=800,results=results)
    print(f"{results_loop}\n")
    print(f"Final Prompt: {Prompt}\n")
    if args.save_json:
        with open(f'/home/YiChen/uq/dev_fig/fig_statistic/llama_token_contributions_{args.metric}_loop_5.json', 'w') as f:
            json.dump(results_loop, f, indent=2)
        print(f"\nSaved to: {args.save_json}")

if __name__ == "__main__":
    main()