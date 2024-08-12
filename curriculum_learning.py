import json
import math
import numpy as np
from collections import defaultdict

def calculate_difficulty(proof_steps):
    if len(proof_steps) == 0:
        return float('inf')  # Assign infinite difficulty to theorems without proofs
    return math.exp(len(proof_steps))

def categorize_difficulty(difficulty):
    if difficulty == float('inf'):
        return "Hard (No proof)"
    elif difficulty <= percentiles[0]:
        return "Easy"
    elif difficulty <= percentiles[1]:
        return "Medium"
    else:
        return "Hard"

file_path = "/raid/adarsh/datasets/pfr_6a5082ee465f9e44cea479c7b741b3163162bb7e/random/train.json"
with open(file_path, 'r') as file:
    data = json.load(file)

# print(json.dumps(data[100], indent=2))
# {'url': 'https://github.com/teorth/pfr', 'commit': '6a5082ee465f9e44cea479c7b741b3163162bb7e', 'file_path': '.lake/packages/mathlib/Mathlib/Algebra/Ring/Subring/Basic.lean', 'full_name': 'Subring.closure_sUnion', 'start': [1024, 1], 'end': [1025, 27], 'traced_tactics': []}
# {'url': 'https://github.com/teorth/pfr', 'commit': '6a5082ee465f9e44cea479c7b741b3163162bb7e', 'file_path': '.lake/packages/mathlib/Mathlib/Order/BooleanAlgebra.lean', 'full_name': 'Bool.inf_eq_band', 'start': [831, 1], 'end': [832, 6], 'traced_tactics': []}
# 100 (1)
# {'url': 'https://github.com/teorth/pfr', 'commit': '6a5082ee465f9e44cea479c7b741b3163162bb7e', 'file_path': '.lake/packages/mathlib/Mathlib/Algebra/Group/Basic.lean', 'full_name': 'one_div_mul_one_div', 'start': [752, 1], 'end': [752, 71], 'traced_tactics': [{'tactic': 'simp', 'annotated_tactic': ['simp', []], 'state_before': 'α : Type u_1\nβ : Type u_2\nG : Type u_3\nM : Type u_4\ninst✝ : DivisionCommMonoid α\na b c d : α\n⊢ 1 / a * (1 / b) = 1 / (a * b)', 'state_after': 'no goals'}]}
# 300 (8)
# {'url': 'https://github.com/teorth/pfr', 'commit': '6a5082ee465f9e44cea479c7b741b3163162bb7e', 'file_path': '.lake/packages/mathlib/Mathlib/Analysis/NormedSpace/OperatorNorm/Mul.lean', 'full_name': 'ContinuousLinearMap.opNorm_lsmul', 'start': [281, 1], 'end': [290, 26], 'traced_tactics': [{'tactic': "refine' ContinuousLinearMap.opNorm_eq_of_bounds zero_le_one (fun x => _) fun N _ h => _", 'annotated_tactic': ["refine' <a>ContinuousLinearMap.opNorm_eq_of_bounds</a> <a>zero_le_one</a> (fun x => _) fun N _ h => _", [{'full_name': 'ContinuousLinearMap.opNorm_eq_of_bounds', 'def_path': './.lake/packages/mathlib/Mathlib/Analysis/NormedSpace/OperatorNorm/Basic.lean', 'def_pos': [180, 9], 'def_end_pos': [180, 28]}, {'full_name': 'zero_le_one', 'def_path': './.lake/packages/mathlib/Mathlib/Algebra/Order/ZeroLEOne.lean', 'def_pos': [26, 15], 'def_end_pos': [26, 26]}]], 'state_before': "𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\n⊢ ‖lsmul 𝕜 𝕜'‖ = 1", 'state_after': "case refine'_1\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nx : 𝕜'\n⊢ ‖(lsmul 𝕜 𝕜') x‖ ≤ 1 * ‖x‖\n\ncase refine'_2\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\n⊢ 1 ≤ N"}, {'tactic': 'obtain ⟨y, hy⟩ := exists_ne (0 : E)', 'annotated_tactic': ['obtain ⟨y, hy⟩ := <a>exists_ne</a> (0 : E)', [{'full_name': 'exists_ne', 'def_path': './.lake/packages/mathlib/Mathlib/Logic/Nontrivial/Defs.lean', 'def_pos': [53, 9], 'def_end_pos': [53, 18]}]], 'state_before': "case refine'_2\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\n⊢ 1 ≤ N", 'state_after': "case refine'_2.intro\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\ny : E\nhy : y ≠ 0\n⊢ 1 ≤ N"}, {'tactic': 'have := le_of_opNorm_le _ (h 1) y', 'annotated_tactic': ['have := <a>le_of_opNorm_le</a> _ (h 1) y', [{'full_name': 'ContinuousLinearMap.le_of_opNorm_le', 'def_path': './.lake/packages/mathlib/Mathlib/Analysis/NormedSpace/OperatorNorm/Basic.lean', 'def_pos': [243, 9], 'def_end_pos': [243, 24]}]], 'state_before': "case refine'_2.intro\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\ny : E\nhy : y ≠ 0\n⊢ 1 ≤ N", 'state_after': "case refine'_2.intro\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\ny : E\nhy : y ≠ 0\nthis : ‖((lsmul 𝕜 𝕜') 1) y‖ ≤ N * ‖1‖ * ‖y‖\n⊢ 1 ≤ N"}, {'tactic': 'simp_rw [lsmul_apply, one_smul, norm_one, mul_one] at this', 'annotated_tactic': ['simp_rw [<a>lsmul_apply</a>, <a>one_smul</a>, <a>norm_one</a>, <a>mul_one</a>] at this', [{'full_name': 'ContinuousLinearMap.lsmul_apply', 'def_path': '.lake/packages/mathlib/Mathlib/Analysis/NormedSpace/OperatorNorm/Mul.lean', 'def_pos': [212, 9], 'def_end_pos': [212, 20]}, {'full_name': 'one_smul', 'def_path': './.lake/packages/mathlib/Mathlib/GroupTheory/GroupAction/Defs.lean', 'def_pos': [483, 9], 'def_end_pos': [483, 17]}, {'full_name': 'NormOneClass.norm_one', 'def_path': './.lake/packages/mathlib/Mathlib/Analysis/Normed/Field/Basic.lean', 'def_pos': [165, 3], 'def_end_pos': [165, 11]}, {'full_name': 'mul_one', 'def_path': './.lake/packages/mathlib/Mathlib/Algebra/Group/Defs.lean', 'def_pos': [480, 9], 'def_end_pos': [480, 16]}]], 'state_before': "case refine'_2.intro\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\ny : E\nhy : y ≠ 0\nthis : ‖((lsmul 𝕜 𝕜') 1) y‖ ≤ N * ‖1‖ * ‖y‖\n⊢ 1 ≤ N", 'state_after': "case refine'_2.intro\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\ny : E\nhy : y ≠ 0\nthis : ‖y‖ ≤ N * ‖y‖\n⊢ 1 ≤ N"}, {'tactic': "refine' le_of_mul_le_mul_right _ (norm_pos_iff.mpr hy)", 'annotated_tactic': ["refine' <a>le_of_mul_le_mul_right</a> _ (norm_pos_iff.mpr hy)", [{'full_name': 'le_of_mul_le_mul_right', 'def_path': './.lake/packages/mathlib/Mathlib/Algebra/Order/Ring/Lemmas.lean', 'def_pos': [236, 9], 'def_end_pos': [236, 31]}]], 'state_before': "case refine'_2.intro\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\ny : E\nhy : y ≠ 0\nthis : ‖y‖ ≤ N * ‖y‖\n⊢ 1 ≤ N", 'state_after': "case refine'_2.intro\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\ny : E\nhy : y ≠ 0\nthis : ‖y‖ ≤ N * ‖y‖\n⊢ 1 * ‖y‖ ≤ N * ‖y‖"}, {'tactic': 'simp_rw [one_mul, this]', 'annotated_tactic': ['simp_rw [<a>one_mul</a>, this]', [{'full_name': 'one_mul', 'def_path': './.lake/packages/mathlib/Mathlib/Algebra/Group/Defs.lean', 'def_pos': [474, 9], 'def_end_pos': [474, 16]}]], 'state_before': "case refine'_2.intro\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\ny : E\nhy : y ≠ 0\nthis : ‖y‖ ≤ N * ‖y‖\n⊢ 1 * ‖y‖ ≤ N * ‖y‖", 'state_after': 'no goals'}, {'tactic': 'rw [one_mul]', 'annotated_tactic': ['rw [<a>one_mul</a>]', [{'full_name': 'one_mul', 'def_path': './.lake/packages/mathlib/Mathlib/Algebra/Group/Defs.lean', 'def_pos': [474, 9], 'def_end_pos': [474, 16]}]], 'state_before': "case refine'_1\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nx : 𝕜'\n⊢ ‖(lsmul 𝕜 𝕜') x‖ ≤ 1 * ‖x‖", 'state_after': "case refine'_1\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nx : 𝕜'\n⊢ ‖(lsmul 𝕜 𝕜') x‖ ≤ ‖x‖"}, {'tactic': 'apply opNorm_lsmul_apply_le', 'annotated_tactic': ['apply <a>opNorm_lsmul_apply_le</a>', [{'full_name': 'ContinuousLinearMap.opNorm_lsmul_apply_le', 'def_path': '.lake/packages/mathlib/Mathlib/Analysis/NormedSpace/OperatorNorm/Mul.lean', 'def_pos': [228, 9], 'def_end_pos': [228, 30]}]], 'state_before': "case refine'_1\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nx : 𝕜'\n⊢ ‖(lsmul 𝕜 𝕜') x‖ ≤ ‖x‖", 'state_after': 'no goals'}]}

difficulties_by_url = defaultdict(list)

for item in data:
    proof_steps = item['traced_tactics']
    difficulty = calculate_difficulty(proof_steps)
    difficulties_by_url[item['url']].append((item['full_name'], item['file_path'], difficulty))

all_difficulties = [diff for url_difficulties in difficulties_by_url.values() for _, _, diff in url_difficulties]

percentiles = np.percentile(all_difficulties, [33, 67])

categorized_theorems = defaultdict(lambda: defaultdict(list))

for url, theorems in difficulties_by_url.items():
    for theorem_name, file_path, difficulty in theorems:
        category = categorize_difficulty(difficulty)
        categorized_theorems[url][category].append((theorem_name, file_path, difficulty))

print("Summary of theorem difficulties by URL:")
for url, categories in categorized_theorems.items():
    print(f"\nURL: {url}")
    for category in ["Easy", "Medium", "Hard", "Hard (No proof)"]:
        theorems = categories[category]
        print(f"  {category}: {len(theorems)} theorems")
        if theorems:
            sorted_theorems = sorted(theorems, key=lambda x: x[2], reverse=True)[:3]
            for name, path, diff in sorted_theorems:
                print(f"    - {name} (File: {path}, Difficulty: {diff:.2f})")

print("\nOverall Statistics:")
total_theorems = sum(len(theorems) for categories in categorized_theorems.values() for theorems in categories.values())
for category in ["Easy", "Medium", "Hard", "Hard (No proof)"]:
    count = sum(len(categories[category]) for categories in categorized_theorems.values())
    percentage = (count / total_theorems) * 100
    print(f"{category}: {count} theorems ({percentage:.2f}%)")