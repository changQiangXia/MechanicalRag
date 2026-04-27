import unittest
from pathlib import Path


class DocumentationExplanationsTest(unittest.TestCase):
    def test_terms_doc_exists_with_expected_sections(self):
        text = Path("docs/terms_and_mechanisms.md").read_text(encoding="utf-8")
        self.assertIn('<a id="control-chain-terms"></a>', text)
        self.assertIn('<a id="execution-trace-fields"></a>', text)
        self.assertIn('<a id="result-stat-fields"></a>', text)
        self.assertIn('<a id="task-examples"></a>', text)
        self.assertIn("## 控制链术语", text)
        self.assertIn("## 执行日志字段", text)
        self.assertIn("## 结果统计字段", text)
        self.assertIn("## 代表性任务示例", text)

    def test_readme_links_to_terms_doc_and_explains_entry_terms(self):
        text = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("### 关键术语说明", text)
        self.assertIn("[控制链术语](docs/terms_and_mechanisms.md#control-chain-terms)", text)
        self.assertIn("[结果统计字段](docs/terms_and_mechanisms.md#result-stat-fields)", text)
        self.assertIn("`belief_state` 是控制器根据检索证据整理出来的对象状态描述", text)
        self.assertIn("`observer-only ablation` 是保留观测日志但关闭在线修复的对照方法", text)

    def test_overview_has_mechanism_and_result_sections(self):
        text = Path("docs/overview.md").read_text(encoding="utf-8")
        self.assertIn("### 3.4 关键机制理解", text)
        self.assertIn("### 3.5 结果字段说明", text)
        self.assertIn("[控制链术语](terms_and_mechanisms.md#control-chain-terms)", text)
        self.assertIn("[结果统计字段](terms_and_mechanisms.md#result-stat-fields)", text)
        self.assertIn("`evidence -> belief -> seed -> solve -> observation -> repair`", text)
        self.assertIn("`observer-only ablation` 用来区分是否真的发生了执行中修正", text)

    def test_simulation_readme_has_variable_and_result_sections(self):
        text = Path("simulation/README.md").read_text(encoding="utf-8")
        self.assertIn("## 关键变量与阶段含义", text)
        self.assertIn("## 结果字段含义", text)
        self.assertIn("[控制链术语](../docs/terms_and_mechanisms.md#control-chain-terms)", text)
        self.assertIn("[执行日志字段](../docs/terms_and_mechanisms.md#execution-trace-fields)", text)
        self.assertIn("[结果统计字段](../docs/terms_and_mechanisms.md#result-stat-fields)", text)
        self.assertIn("`belief_state` 负责把检索证据整理成后续控制阶段可直接使用的状态描述", text)
        self.assertIn("`online_diagnosis_count` 记录有多少次 trial 进入了执行中的诊断或修正路径", text)


if __name__ == "__main__":
    unittest.main()
