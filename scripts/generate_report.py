from pathlib import Path
import shutil

def generate_docs_results() -> None:
    reports_dir = Path("reports")
    docs_path = Path("docs/results.md")
    summary_path = reports_dir / "summary.md"

    if not summary_path.exists():
        raise FileNotFoundError("No se encontró reports/summary.md. Ejecuta el pipeline antes.")

    # Copiar imágenes al directorio docs/
    for img in ["equity_curve.png", "drawdown_curve.png", "rolling_sharpe.png", "returns_hist.png"]:
        src = reports_dir / img
        dst = Path("docs") / img
        if src.exists():
            shutil.copy(src, dst)
            print(f"🖼️ Copiado {img} a docs/")

    # Leer contenido del summary
    content = summary_path.read_text(encoding="utf-8")

    # Crear página docs/results.md
    docs_md = f"""# 📊 Resultados Recientes

Este informe se genera automáticamente a partir de la última ejecución del pipeline (`quantml.pipelines.research`).

---

{content}
"""
    docs_path.write_text(docs_md, encoding="utf-8")
    print(f"✅ Documento actualizado: {docs_path}")


if __name__ == "__main__":
    generate_docs_results()
