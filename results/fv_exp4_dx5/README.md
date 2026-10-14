# FV tham chiếu exp4, nền phẳng không bột, lưới 5 µm

Cấu hình: `run()` trong `lpbfpinn/fv_stage0.py` với LPBF_PROC=exp4 (120 W, 0.6 m/s, r_b = 71 µm),
tính chất hằng số, không bột, không bay hơi, nguồn thể tích Beer–Lambert, dx = 5 µm.

Lệnh chạy (script nằm ngoài repo, nội dung tương đương):

    LPBF_PROC=exp4 PYTHONPATH=. python3 -c "import json; from lpbfpinn.fv_stage0 import run; r=run(dx=5e-6); r['dx_m']=5e-6; open('results/fv_exp4_dx5/diag.json','w').write(json.dumps(r, indent=2))"

Trạng thái: đã xong. Kết quả: cân bằng năng lượng 2.9·10⁻¹⁴; T_max tại x = 0 là 2434 K (20 µm: 2460 K, 10 µm: 2354 K). Kết quả không đơn điệu theo lưới, nên tham chiếu chưa hội tụ (xem README mục hội tụ trong bản báo cáo). So sánh với `results/fv_exp4_dx10/` và `results/fv_exp4_dx20/`
để kiểm tra hội tụ theo lưới.
