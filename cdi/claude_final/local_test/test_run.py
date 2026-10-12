"""本機測試 --run 的路徑:把「考試資料」的 seed 區間換到 99,000,000–99,999,999(未使用),不產生真正的期末考資料。
用法:cd stage && CDI_QUICK=1 CDI_RUN_ROOT=../runs python ../test_run.py <0..16 | final>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
arg = sys.argv[1]
sys.argv = ["scan_final.py", "--run", arg]
import scan_final as sf                                               # noqa: E402

sf.EXAM_AB = dict(field=99_000_000, noise=99_100_000, pos=99_200_000, df=99_300_000)
sf.EXAM_C = dict(obj=99_500_000, noise=99_600_000, pos=99_700_000, df=99_800_000, newdef=99_900_000, init=99_950_000)
sf.EXAM_LO, sf.EXAM_HI = 99_000_000, 99_999_999
print("⚠️ 測試:考試資料的區間換成 99M(本機、QUICK)")
sf.main()
