import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import List, Dict, Any


def load_model(model_path: str, label: str):
    print(f"加载{label}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print(f"{label}加载完成\n")
    return model, tokenizer

def test_nli(model, tokenizer, premise, hypothesis):
    """测试NLI任务"""
    instruction = f"判断以下两个句子之间的逻辑关系（蕴含、矛盾或中立）：\n前提：{premise}\n假设：{hypothesis}"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=False
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def evaluate_cases(model, tokenizer, cases: List[Dict[str, str]], title: str) -> Dict[str, Any]:
    records = []
    correct = 0

    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

    for case in cases:
        note = case.get("note", "样本")
        print(f"\n{note}")
        print(f"前提: {case['premise']}")
        print(f"假设: {case['hypothesis']}")
        print(f"期望: {case['expected']}")

        response = test_nli(model, tokenizer, case["premise"], case["hypothesis"])
        print(f"回答: {response}")

        is_correct = case["expected"] in response
        print("结果: 正确" if is_correct else "结果: 错误")

        correct += int(is_correct)
        records.append({
            "note": note,
            "expected": case["expected"],
            "response": response.strip(),
            "is_correct": is_correct
        })

    accuracy = correct / len(cases) if cases else 0.0
    print(f"\n准确率: {accuracy*100:.1f}%")
    return {"records": records, "accuracy": accuracy}


def save_results_markdown(items: List[Dict[str, Any]], output_path: str):
    path = Path(output_path)
    lines: List[str] = ["# 模型测试结果", ""]

    for item in items:
        lines.append(f"## {item['label']}")
        lines.append(f"英文NLI准确率: {item['nli']['accuracy']*100:.1f}%")
        lines.append(f"中文泛化准确率: {item['general']['accuracy']*100:.1f}%")
        lines.append("")

        lines.append("英文NLI明细:")
        for record in item["nli"]["records"]:
            lines.append(f"- {record['note']} | 期望: {record['expected']} | 回答: {record['response']} | 结果: {'正确' if record['is_correct'] else '错误'}")
        lines.append("")

        lines.append("中文测试明细:")
        for idx, record in enumerate(item["general"]["records"], 1):
            lines.append(f"- 中文样本{idx} | 期望: {record['expected']} | 回答: {record['response']} | 结果: {'正确' if record['is_correct'] else '错误'}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown结果已保存到 {path.resolve()}")


def main():
    base_model_path = "model"
    sft_model_path = "output"  
    report_path = "model_test_results.md"

    test_cases = [
        {
            "premise": "Dana Reeve, the widow of the actor Christopher Reeve, has died of lung cancer at age 44.",
            "hypothesis": "Christopher Reeve had an accident.",
            "expected": "矛盾",
            "note": "验证集样本1"
        },
        {
            "premise": "Yet, we now are discovering that antibiotics are losing their effectiveness against illness.",
            "hypothesis": "Bacteria is winning the war against antibiotics.",
            "expected": "蕴含",
            "note": "验证集样本2"
        },
        {
            "premise": "In 1979, the leaders signed the Egypt-Israel peace treaty on the White House lawn.",
            "hypothesis": "The Israel-Egypt Peace Agreement was signed in 1979.",
            "expected": "蕴含",
            "note": "验证集样本3"
        },
        {
            "premise": "Oil prices fall back as Yukos oil threat lifted",
            "hypothesis": "Oil prices rise.",
            "expected": "矛盾",
            "note": "验证集样本4"
        },
        {
            "premise": "Steve Jobs was attacked by Sculley and other Apple executives and resigned from the company.",
            "hypothesis": "Steve Jobs worked for Apple.",
            "expected": "蕴含",
            "note": "验证集样本5"
        }
    ]

    general_test = [
        {
            "premise": "今天天气很好，阳光明媚。",
            "hypothesis": "今天在下雨。",
            "expected": "矛盾"
        },
        {
            "premise": "他是一名医生。",
            "hypothesis": "他在医院工作。",
            "expected": "蕴含或中立"
        }
    ]

    results = []

    base_model, base_tokenizer = load_model(base_model_path, "基础模型")
    base_nli = evaluate_cases(base_model, base_tokenizer, test_cases, "基础模型 - 英文NLI")
    base_general = evaluate_cases(base_model, base_tokenizer, general_test, "基础模型 - 中文测试")
    results.append({"label": "基础模型", "nli": base_nli, "general": base_general})

    sft_model, sft_tokenizer = load_model(sft_model_path, "SFT模型")
    sft_nli = evaluate_cases(sft_model, sft_tokenizer, test_cases, "SFT模型 - 英文NLI")
    sft_general = evaluate_cases(sft_model, sft_tokenizer, general_test, "SFT模型 - 中文测试")
    results.append({"label": "SFT模型", "nli": sft_nli, "general": sft_general})

    save_results_markdown(results, report_path)


# if __name__ == "__main__":
#     main()

