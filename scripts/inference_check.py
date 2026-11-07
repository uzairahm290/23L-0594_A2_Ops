"""Task 4.1 - load the champion once through the web app's inference module and classify
known images.   python scripts/inference_check.py <threshold> img1 img2 ..."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "web")]
from app.inference import LocalPredictor  # noqa: E402

t0 = time.perf_counter()
p = LocalPredictor(str(ROOT / "models/champion/model.keras"), float(sys.argv[1]), "23L-0594_A2_Ops_visionguard", "champion")
print(f"model loaded once in {time.perf_counter() - t0:.1f}s")
for f in sys.argv[2:]:
    r = p.predict(Path(f).read_bytes())
    print(f"{Path(f).name:<30} {r.label:<13} confidence={r.confidence:.1%}  p(AI)={r.p_ai:.4f}  {r.latency_ms} ms")
