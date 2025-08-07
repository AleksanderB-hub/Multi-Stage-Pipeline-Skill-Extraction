import json
import logging
import random
from typing import List, Dict
from openai import OpenAI
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are an expert skill classifier. "
    "Given a sentence and a list of possible skills, your task is to select only the skills that are explicitly or implicitly required. "
    "Be precise and avoid including unrelated or weakly related skills. "
    "Return a JSON **object** of the form "
    "{\"relevant_skills\": [\"skill1\", \"skill2\", ...]}. "
    "If no skills are relevant, return {\"relevant_skills\": []}. "
    "Do not add any other keys or text."
)

def get_demonstration(skills: List[str], reference_data: List[Dict]) -> Dict:
    if not reference_data:
        return None

    ref_set = set(skills)
    best_examples = []
    max_overlap = -1

    for ex in reference_data:
        overlap = len(ref_set & set(ex["candidate_labels"]))
        if overlap > max_overlap:
            best_examples = [ex]
            max_overlap = overlap
        elif overlap == max_overlap:
            best_examples.append(ex)

    if max_overlap == 0:
        return random.choice(reference_data)

    return random.choice(best_examples)

def run_llm_extraction(api_key: str, test_data: List[Dict], reference_data: List[Dict], model: str, use_demo: bool) -> List[Dict]:
    client = OpenAI(api_key=api_key)

    results, n_errors = [], 0
    for item in tqdm(test_data, desc="LLM tagging"):
        payload = {
            "sentence": item["sentence"],
            "skills": item["candidate_labels"],
        }

        if use_demo and reference_data is not None:
            demo = get_demonstration(item["candidate_labels"], reference_data)
            payload.update({
                "demo_sentence": demo["sentence"],
                "demo_skills": demo["candidate_labels"],
                "demo_answer": demo["true_labels"],
            })

        user_prompt = json.dumps(payload, ensure_ascii=False)

        try:
            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )

            resp_json = json.loads(response.choices[0].message.content)

            if isinstance(resp_json, list):
                selected = resp_json
            elif isinstance(resp_json, dict):
                selected = resp_json.get("relevant_skills", [])
            else:
                logging.warning("Unexpected JSON type %s", type(resp_json))
                selected = []

        except Exception as e:
            n_errors += 1
            logging.exception("LLM call failed")
            selected = []

        results.append({**item, "extracted": selected})

    logging.info("Completed with %d errors", n_errors)
    return results

def evaluate_skill_extraction_v2(results: List[Dict], predicted_field: str = "extracted", gold_field: str = "true_labels") -> Dict[str, float]:
    all_preds, all_trues = [], []

    for item in results:
        pred_set = set(item.get(predicted_field, []))
        true_set = set(item.get(gold_field, []))

        all_labels = pred_set.union(true_set)
        for lab in all_labels:
            all_preds.append(1 if lab in pred_set else 0)
            all_trues.append(1 if lab in true_set else 0)

    precision = precision_score(all_trues, all_preds, zero_division=0)
    recall = recall_score(all_trues, all_preds, zero_division=0)
    f1 = f1_score(all_trues, all_preds, zero_division=0)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }
