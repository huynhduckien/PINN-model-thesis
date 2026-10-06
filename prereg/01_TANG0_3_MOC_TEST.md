# Đăng ký trước 01 — các mốc test tầng 0–3 (trước khi huấn luyện chính)

Ngày: 2026-10-06. Viết **trước** khi viết code của các mốc test và trước khi chạy bất kỳ mốc nào. Commit đăng ký này trước, mọi thay đổi sau đó phải qua một đăng ký sửa đổi có ngày và lý do. Làm theo `CLAUDE.md` (nghĩ trước, đơn giản, thay đổi tối thiểu, tiêu chí kiểm chứng được).

## 1. Quyết định của người dùng (2026-10-06)

- **D0 = hướng B:** giữ 300 W, khôi phục bay hơi, nền vệt thường là `B3_84d0289e` (bỏ S1). Điều này **trái** quyết định cũ "không bay hơi"; lý do ghi trong `00_KE_HOACH_CHAY_OVERHANG.md` §2: ở 300 W thông lượng bay hơi là 11% cường độ hấp thụ ở 4,000 K và 92% ở 5,000 K, mà mô hình không bay hơi cho T_peak khoảng 8,200 K.
- **Hiệu chỉnh:** chia theo **đại lượng** (hiệu chỉnh hấp thụ theo độ sâu, kiểm bằng độ rộng). Giữ một điều kiện (300 W, 0.8 m/s).
- Thư mục mới `lpbfpinn/`, `git init`.
- Các ngưỡng đề xuất ở kế hoạch được chấp nhận: tối ưu tốc độ chỉ khi đạo hàm bậc hai > 50% thời gian bước; ổn định M3 là 5 µm cho kích thước và 3 µm cho Δ độ sâu; trần 75 phút mỗi bước huấn luyện.
- Giữ DeepXDE, `lpbf_pinn.py` đóng băng trong `legacy/`; không FD; không dữ liệu trong loss.

## 2. Cấu hình cố định (chép nguyên từ `stepB_phase.ipynb`, đã dùng cho B3)

IN718 hiệu dụng ρ = 8000, c_p = 600, k = 22, T_amb = 298; T_sol = 1533, T_liq = 1609, L_f = 272 kJ/kg, T_vap = 3120; bột D50 30 µm, độ xốp 0.45, α_s = 0.35, k_khí 0.042, iface 15 µm; 300 W, 0.8 m/s, r_b 40 µm, ramp 10 µs; miền 1500 × 380 × 240 µm (vệt thường, đúng B3) hoặc 1500 × **480** × 240 µm (nêm, như P1); mạng 6×24 sin; ẩn nhiệt σ = 150 K; k lỏng f = 2.8 tâm T_liq + 2σ; bay hơi Fang (L_v 2.224e6, M 0.166534, p_a 1 atm, T_cap 6,000 K). Điểm lấy mẫu như `SAMPLING` của Bước A (2^16 + 2^18 + 2^12 + 2^15).

Nêm: 45°, đỉnh nêm **dưới tâm chùm (y_a = 0)**. Lỗi y_a = +20 µm đã ghi ở `coding/scratchpad/OVERHANG_FD_PREREG.md`.

## 3. Quy tắc chung (khóa)

1. Chạy theo thứ tự tầng 0 → 1 → 2 → 3. **Trượt một mốc thì dừng, báo cáo, không chạy mốc sau.** Ngoại lệ: S0.8 chỉ ảnh hưởng M5, không chặn các mốc khác.
2. Mỗi tiến trình GPU phải giữ khóa `.run.lock`; tiến trình thứ hai từ chối chạy. Không chạy phân tích nặng song song với một lượt huấn luyện.
3. Trước khi can thiệp tiến trình nền: kiểm `date`, PID, `nvidia-smi` và mtime của log (không chỉ đọc log).
4. Mọi file sinh ra của các mốc test nằm trong `runs/` (bị git bỏ qua) hoặc thư mục tạm; `legacy/ckpt/` không bị ghi (kiểm hash trước và sau).
5. Không đổi tiêu chí sau khi thấy kết quả. Sai sót của mốc test (không phải của PINN) được sửa bằng đăng ký sửa đổi, như đã làm ở R0.
6. Trần thời gian: tầng 0 ≤ 20 phút, tầng 1 ≤ 20 phút, tầng 2 ≤ 15 phút, tầng 3 ≤ 20 phút (GPU, không kể thời gian viết code). Vượt 1.5× trần thì dừng và báo cáo.

## 4. Tầng 0 — tĩnh, không huấn luyện

| ID | Kiểm | Tiêu chí |
|---|---|---|
| S0.1 | Môi trường: phiên bản torch, deepxde, GPU; **không** có tiến trình python khác dùng GPU | GPU còn trống ≥ 14 GB; không tiến trình lạ |
| S0.2 | Tag tính lại từ `config.py` của `A1s`, `B1`, `B2`, `B3` | bằng `A1s_f831590d`, `B1_8ff624dd`, `B2_d610497f`, `B3_84d0289e`; file `.pt` có; `lpbf_pinn.py` trong `legacy/` trùng hash bản gốc `b0f5955a2385e79c...` |
| S0.3 | **Canary:** nạp `B3_84d0289e`, đo bể chảy (`box`, `melt_pool`) và `probe` bằng chính các hàm của `legacy/lpbf_pinn.py` | rộng 166 ± 2, sâu 90 ± 2, dài 516 ± 4 µm (số PROGRESS §5.4, cùng phép đo snapshot); (tích + bay hơi)/nạp 0.805 ± 0.01; năng lượng do residual trong [−10, −5] mJ (số đã ghi −8.4 và −7.4) |
| S0.4 | `indicator.py` áp lên trường giải tích R0 (`coding/scratchpad/overhang_r0.py`, 59 s) | dài/rộng/sâu trùng R0 (520.2/122.0/85.9 µm) ± 2 µm; độ xuyên dưới đáy lớp bột = sâu − 60 µm = 25.9 ± 2 µm |
| S0.5 | Hình học nêm y_a = 0: bản đồ vật liệu `s(y, z)` bằng `mat_fields` với `wedge` | `s` = 0.5 tại z = 60 µm ở y = 120 ± 10 µm; diện tích vùng s < 0.5 trong z ≤ 180 µm, y ∈ [−240, 240] bằng 27,000 µm² ± 3%; (y, z) = (200, 60) µm là đặc (s ≤ 0.01); (−100, 60) µm là bột (s ≥ 0.99); nêm không vượt qua lớp bột (s ≥ 0.99 ở z ≥ 195 µm, mọi y) |
| S0.6 | MMS có nêm: `check_pde(sc, path)` với `wedge` và vật lý B3 | lệch lớn nhất / RMS ≤ 1e-10 |
| S0.7 | `check_config`, `check_source`, `check_density`, `check_data` cho cấu hình nêm B3 | không [WARN] |
| S0.8 | **Không GPU, chỉ cho M5:** tải được **bảng số** thực nghiệm (P, v, rộng, sâu) của Scime & Beuth 2019 và NIST AM Bench 2022 (AMB2022-03); lọc các điều kiện IN718 có sâu/nửa rộng ≲ 1.2 | ≥ 2 điều kiện IN718 trong dải 200–300 W, 0.6–1.2 m/s có bảng số rộng và sâu. Không đạt thì M5 dừng ở "dùng hấp thụ đo, báo độ lệch" (không chặn gì khác). Không dùng dữ liệu của bản thảo JOLT. |

## 5. Tầng 1 — chi phí (GPU, không cập nhật trọng số)

| ID | Kiểm | Tiêu chí |
|---|---|---|
| S1.1 | Đo 30 bước Adam toàn lô trên cấu hình nêm B3: giây/bước và bộ nhớ GPU đỉnh | ghi nhận; nếu > 0.25 s/bước (số đã đo 0.13) thì **dừng** để tìm nguyên nhân |
| S1.2 | Loss khởi điểm [L_PDE, L_top] của B3 trên hình học nêm | chỉ ghi nhận (B3 từng bắt đầu ở 18.1 và vẫn đạt; B1→B1k bắt đầu ở 14.95 và trượt; không phải ngưỡng cứng) |
| S1.3 | Profile một bước Adam (`torch.profiler`): thời gian cho autograd bậc hai trong `pde`, mạng, phần còn lại | ghi nhận; **chỉ làm tối ưu tốc độ nếu đạo hàm bậc hai chiếm > 50% thời gian bước** |

## 6. Tầng 2 — khói đầu-cuối trên ngân sách tí hon

Chạy trên `runner.py` và `gates.py` (chưa có; sẽ viết sau khi đăng ký này được duyệt).

| ID | Kiểm | Tiêu chí |
|---|---|---|
| S2.1 | Adam 20 + L-BFGS float64 khối 5 vòng (tối đa 12) + resume + đánh giá; mọi file vào thư mục tạm | chạy hết đường tới `run.json`; hash `legacy/ckpt/` không đổi; tag khác tag thật |
| S2.2 | **Cổng vật lý trên checkpoint đã có** (bản cổng nhẹ; so với `probe` đầy đủ) | `B3_84d0289e`: **PASS**. `B1k_4d5bf24c`: **FAIL**. `B1kc_e7de526d`: **FAIL**. `B1f1.4_bd84c1f5`: **FAIL ở gate E_res** (+15.3 mJ) và **đạt gate T_min**. Cổng nhẹ phải cho cùng kết luận từng gate với `probe` đầy đủ và E_res nhẹ trong 3 mJ so với đầy đủ |
| S2.3 | Giết tiến trình (kill) giữa một khối rồi `--resume` | mất ≤ 1 khối; kết quả cuối khối tiếp theo hợp lệ; giám sát phát hiện tiến trình chết trong ≤ 3 phút |
| S2.4 | Khởi chạy hai tiến trình GPU cùng lúc | tiến trình thứ hai từ chối ngay với thông báo rõ |

Định nghĩa cổng (khóa): PASS khi (a) |năng lượng do residual| ≤ 10 mJ, (b) T hụt dưới T_amb ≤ 1% biên độ, (c) (tích + bay hơi)/nạp ≤ 1. E_res tính trên 2^16 điểm cố định (seed 101, như `probe`); T_min trên 2^18 điểm (nhẹ) hoặc 2^21 điểm (đầy đủ).

## 7. Tầng 3 — canary huấn luyện ngắn

| ID | Kiểm | Tiêu chí |
|---|---|---|
| S3.1 | Bậc `B3W0` (nêm y_a = 0, miền 480 µm, vật lý B3), khởi động từ `B3_84d0289e`, Adam 500 bước lr 1e-4, đo cổng nhẹ mỗi 100 bước | không NaN/Inf; tổng loss ở bước 500 ≤ 0.7 × loss khởi điểm **[ĐỀ XUẤT]**; giây/bước ≤ 0.25; ghi E_res, T_min (không phải điều kiện đạt) |

## 8. Sau khi qua tầng 3

Viết đăng ký 02 (M2 `B3W0`: ngân sách, cổng dừng, tiêu chí, dự đoán) **trước** khi chạy M2. Đăng ký 02 phải ghi: số đo S1.1–S1.3 và S3.1 đã dùng để chọn ngân sách; trần 75 phút; M3 (ổn định theo tập điểm, 50/70/90% thời gian, gấp đôi ngân sách) với ngưỡng 5 µm và 3 µm; M4 (chỉ báo overhang); M5 (hiệu chỉnh theo đại lượng).

## 9. Dự đoán (ghi trước, có thể sai)

- S0.2–S0.7 đạt ngay lượt đầu: khoảng 60%. Rủi ro lớn nhất: chép cấu hình (một hằng số lệch làm tag đổi), và định nghĩa phép đo trong S0.3 (số cũ lấy từ hai chỗ ghi khác nhau trong PROGRESS).
- S0.3 đạt: khoảng 75%.
- S0.8 đạt: khoảng 50% (chưa biết có bảng số tải được).
- S1.1 ≤ 0.25 s/bước: khoảng 90%. S1.3 cho đạo hàm bậc hai > 50%: khoảng 60% (suy từ cấu trúc `pde` có hai lần `autograd.grad` bậc hai trên 3 hướng).
- S2.2 phân biệt đúng bốn checkpoint: khoảng 70%. Chỗ dễ sai: `B1f1.4` ở sát ngưỡng E_res (15.3 so với 10 mJ), nhưng chiều của kết luận không đổi.
- S3.1 đạt: khoảng 70%.
- Tổng thời gian GPU tầng 0–3: 40–60 phút.

## 10. Giới hạn (khóa)

Các mốc test chỉ kiểm hạ tầng và tính khả thi, không kiểm vật lý của overhang. Chúng không thay thế đăng ký 02. Nghiệm giải tích R0 (S0.4) chỉ có cho vật liệu đồng nhất; cổng ở S2.2 hiệu chuẩn trên bốn checkpoint của cùng một họ vật lý, không chứng minh cổng đúng cho hình học nêm.
