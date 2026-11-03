import os
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(ROOT_DIR, "artifacts", "tables")

def export_consolidated_report():
    report_path = os.path.join(TABLES_DIR, "consolidated_evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as out:
        out.write("# CLARIS: Dossiê de Evidências Experimentais e Validação Teórica\n\n")
        out.write("Compilação automática de tabelas e métricas para suporte à escrita do artigo.\n\n")

        files = [
            ("table1_latency.csv", "Tabela 1: Latência Computacional por Estágio (ms)"),
            ("table2_grounding_adversarial.csv", "Tabela 2: Auditoria de Grounding e Testes Adversariais (H1)"),
            ("table3_drift_sensitivity.csv", "Tabela 3: Sensibilidade Dual de Detecção de Drift (H3)"),
            ("table4_case_study_deltas.csv", "Tabela 4: Deltas Determinísticos do Caso de Estudo"),
            ("table5_baseline_comparison.csv", "Tabela 5: Comparativo Estrutural contra Baselines da Literatura")
        ]

        for filename, title in files:
            p = os.path.join(TABLES_DIR, filename)
            if os.path.exists(p):
                df = pd.read_csv(p)
                out.write(f"## {title}\n\n")
                out.write(df.to_markdown(index=False))
                out.write("\n\n---\n\n")

    print(f"✓ Dossiê consolidado exportado com sucesso em:\n  {report_path}")

if __name__ == "__main__":
    export_consolidated_report()