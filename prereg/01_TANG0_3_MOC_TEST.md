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

---

## Kết quả tầng 0 (2026-10-06 12:57): **ĐẠT** cả bảy mốc S0.1–S0.7, 27 giây, không huấn luyện

Mã: `config.py`, `indicator.py`, `smoke/tier0.py`; kết quả máy đọc `runs/tier0/report.json` (không commit, `runs/` bị bỏ qua). `legacy/ckpt` không đổi (kiểm hash đầu và cuối).

| Mốc | Kết quả | Tiêu chí |
|---|---|---|
| S0.1 | torch 2.5.1+cu121, deepxde 1.15.0, GPU 381/16,376 MiB, không tiến trình liên quan khác | GPU trống ≥ 14 GB |
| S0.2 | hash `lpbf_pinn.py` `b0f5955a2385e79c`; tag `A1s_f831590d`, `B1_8ff624dd`, `B2_d610497f`, `B3_84d0289e` đúng; có file `.pt` | khớp |
| S0.3 (canary B3) | rộng/sâu/dài **166.0 / 90.0 / 516.0 µm**; (tích + bay hơi)/nạp **0.805**; E_res **−7.4 mJ**; T_peak 5,555 K; T_min 263.1 K | 166±2 / 90±2 / 516±4; 0.805±0.01; [−10, −5] |
| S0.4 | `indicator` trên nghiệm giải tích R0: dài/rộng/sâu 520.15/121.99/85.95 µm, lệch +0.00 mỗi chiều; độ xuyên 25.95 µm | ±2 µm |
| S0.5 | nêm y_a = 0: s = 0.5 tại z = 60 µm ở y = 120.0 µm; diện tích vùng đặc 26,877 µm² (−0.46%); s(200,60) = 7.9e-14; s(−100,60) = 1.0000; s nhỏ nhất ở z = 195 µm là 0.9997 | 120±10; ±3%; ≤0.01; ≥0.99; ≥0.99 |
| S0.6 | MMS có nêm (ẩn nhiệt + k lỏng + bay hơi): 1.28e-14 | ≤ 1e-10 |
| S0.7 | `check_config`, `check_source`, `check_density`, `check_data` cho nêm: 0 cảnh báo; 364,544 điểm neo, 37,079 điểm mặt trên | 0 [WARN] |

**Ghi chú thủ tục (phải nêu rõ):** `config.py`, `indicator.py`, `smoke/tier0.py` và `legacy/overhang_r0.py` đã được ghi vào thư mục này lúc 12:55–12:56, và tầng 0 được chạy lúc 12:56:24, **không phải bởi phiên làm việc đang viết đăng ký này** (người dùng đã duyệt và bảo bắt đầu viết code; chưa xác nhận phiên nào đã viết và chạy). Tôi đã đọc toàn bộ `config.py` và `tier0.py` và phần còn lại của `indicator.py` trước khi chấp nhận kết quả: các tiêu chí khớp đăng ký bản khóa; `half_extent` của `indicator.py` cài khác R0 (vòng lặp từng hàng, thay vì vector hóa) nên S0.4 là đối chiếu độc lập; `legacy/overhang_r0.py` giống hệt `coding/scratchpad/overhang_r0.py` (diff rỗng). Bảng băm `legacy/SHA256SUMS.txt` do tôi tạo.

**Chấm dự đoán:** tầng 0 đạt ngay lượt đầu (60%): đúng. S0.3 đạt (75%): đúng. S0.8 (50%): chưa làm.

**Còn mở:** S0.8 (bảng số thực nghiệm, không GPU, chỉ chặn M5) chưa thực hiện; tầng 1–3 chưa chạy.

---

## 11. Ghi chú nguồn gốc (sửa 2026-10-06 13:00; thay một bản sao trùng của bảng kết quả tầng 0)

Trong cùng thư mục có **hai phiên làm việc Claude chạy song song** (tiến trình VS Code mở lúc 12:55 và phiên chính). Phiên viết mã và **chạy tầng 0 (12:56:24) rồi tầng 1 (12:57)** là phiên đang ghi các mục 12–13; phiên còn lại đọc kết quả, commit `11e2f22` và thêm bảng "Kết quả tầng 0" ở trên (số liệu trùng với `runs/tier0/report.json`; nó ghi rằng không biết phiên nào viết mã, thông tin đó là sai: `config.py`, `indicator.py`, `lock.py`, `smoke/tier0.py`, `smoke/tier1.py`, `legacy/overhang_r0.py` do phiên chạy tầng 0 viết). Bản sao trùng của bảng ở đây đã bị xóa; không số liệu nào bị đổi. Hai phiên dùng chung GPU 16 GB; khóa `lock.py` ngăn hai tiến trình GPU chạy cùng lúc, nhưng không ngăn hai phiên cùng sửa file và commit. Quyết định để một phiên duy nhất làm việc trên thư mục này thuộc về người dùng.

## 12. Kết quả tầng 1 (2026-10-06 12:57): **ĐẠT** (S1.1 đạt; S1.2 và S1.3 chỉ ghi nhận)

Mã `smoke/tier1.py` (giữ khóa GPU `lock.py`); kết quả `runs/tier1/report.json`. Cấu hình B3W0 (nêm y_a = 0, Ly = 480 µm), tag `B3W0_fff2a3d4`, 364,544 điểm neo; trọng số B3 nạp vào, không lưu gì.

| ID | Kết quả |
|---|---|
| S1.1 | Adam toàn lô qua DeepXDE: **0.140 s/bước**, bộ nhớ GPU đỉnh 6.7 GB (ngưỡng 0.25) |
| S1.2 | Loss khởi điểm của B3 trên nêm: **L_PDE 0.332, L_top 8.8e-5** (B3 từng bắt đầu ở 18.1; B1→B1k ở 14.95). Rất thuận lợi so với các lượt trước |
| S1.3 | Cùng 364,544 điểm, vòng thủ công (không DeepXDE): chỉ lan truyền mạng 2 ms; bước chỉ dùng đạo hàm bậc một (xấp xỉ) **15 ms**; bước đầy đủ (đạo hàm bậc hai + backward) **108 ms**. Phần thời gian do đạo hàm bậc hai ≈ **86%** > ngưỡng 50% → theo đăng ký **làm** tối ưu bậc hai |

Hai điều cần ghi trung thực:
- Chênh DeepXDE (140 ms) − bước thủ công đầy đủ (108 ms) = **33 ms**. Tôi đã nói trước đó "chi phí DeepXDE ≈ 0"; con số đó chỉ đo `train_next_batch`. 33 ms này **chưa quy được cho DeepXDE**, vì bước thủ công chưa gồm loss mặt trên (37,079 điểm có đạo hàm bậc một và bay hơi) và bước tối ưu.
- 86% là proxy (so với bước bậc một), không phải profiler thật; dùng để quyết định hướng, không phải con số chính xác.

Chấm dự đoán: S1.1 ≤ 0.25 s/bước (90%): đúng. Đạo hàm bậc hai > 50% (60%): đúng (86%).

## 13. Sửa đổi 01a (2026-10-06, trước khi viết `fastpde.py`): tối ưu bậc hai

Căn cứ: S1.3. Mục tiêu: tính residual PDE **chính xác như `lpbf_pinn.pde`** nhưng không dùng đạo hàm bậc hai bằng autograd, để giảm thời gian mỗi bước. Phương pháp: lan truyền thuận giá trị, đạo hàm bậc một (x, y, z, t) và đạo hàm bậc hai chéo (xx, yy, zz) qua từng lớp `sin` của mạng 6×24 (quy tắc chuỗi giải tích: (sin z)' = cos z·z', (sin z)'' = −sin z·z'² + cos z·z''); mặt nạ cứng M(x) (chỉ phụ thuộc x, không phụ thuộc trọng số) lấy đạo hàm bằng autograd riêng (không cần đồ thị qua tham số), rồi θ = M·y bằng quy tắc tích. Divergence dùng dạng khai triển (đã kiểm ở `check_pde`): div = κ_s·K·Σθ_jj + κ_s·K′·Σθ_j² + K·(∇ₓκ_s·∇θ), κ_s là độ dẫn tĩnh theo (y, z). Không hỗ trợ `sc['pm']`.

Tiêu chí khóa (tất cả phải đạt trước khi dùng cho bất kỳ lượt huấn luyện nào):

| ID | Kiểm | Tiêu chí |
|---|---|---|
| E1 | Residual tại 2^14 điểm ngẫu nhiên (cả trong miền và dải mặt phân giới/nêm), float64, trọng số B3 | max\|r_nhanh − r_cũ\| / RMS(r_cũ) ≤ 1e-10 |
| E2 | Gradient của loss PDE theo trọng số, float64 | sai tương đối chuẩn L2 ≤ 1e-8 |
| E3 | Loss PDE tại đủ 364,544 điểm neo, float32, trọng số B3 trên nêm | sai tương đối ≤ 1e-4 so với 0.3316 (S1.2) và so với `pde` cũ |
| E4 | Tốc độ: bước (PDE loss + backward, vòng thủ công, 364,544 điểm) | ≤ 72 ms (nhanh ≥ 1.5× so với 108 ms); bộ nhớ đỉnh ≤ 12 GB |
| E5 | Tích hợp DeepXDE: truyền hàm residual mới làm `pde` của `TimePDE`; 30 bước Adam; loss ở bước 0 | loss bước 0 trùng [0.3316, 8.8e-5] sai ≤ 1e-4 tương đối; s/bước được báo cáo |

Quy tắc: E1–E3 trượt thì **bỏ** `fastpde`, giữ `pde` cũ và báo cáo. E4 trượt thì giữ `pde` cũ (không có lợi), nhưng giữ mã làm bản ghi. Không chạy huấn luyện trong sửa đổi này; nó chỉ kiểm tương đương và tốc độ. Mã trong `lpbfpinn/fastpde.py`, mốc test `smoke/tier1b.py`.

Dự đoán (ghi trước): E1–E3 đạt: khoảng 80% (rủi ro: lỗi đạo hàm mặt nạ hoặc điểm không khả vi của `max` ở nêm). E4 đạt (≥ 1.5×): khoảng 70%; mức tăng tốc kỳ vọng 2–4×, không chắc. E5 đạt: 85% nếu E1–E4 đạt.

---

## 12a. Bổ sung của phiên thứ hai (2026-10-06 13:00): tách chi phí DeepXDE trong S1.3

(Thay một bản kết quả tầng 1 trùng với mục 12; chỉ giữ thông tin mới. Mã: `smoke/tier1b_overhead.py`, chạy dưới khóa GPU, kết quả `runs/tier1/overhead.json`; **không phải mốc có ngưỡng**.)

Mục 12 ghi chênh DeepXDE (140 ms) − bước thủ công (108 ms) = 33 ms là chưa quy được cho thư viện. Đo tách: bước DeepXDE theo hiệu hai độ dài chạy (loại phần cố định 0.229 s mỗi lần `train`): **0.132 s**; bước thủ công chỉ L_PDE: 0.107 s; **L_PDE + L_top (37,079 điểm): 0.109 s** (L_top chỉ thêm 2 ms). Chênh còn lại **23 ms = 17%** bước DeepXDE. Đây là **cận trên** của chi phí thư viện: bước thủ công không có cập nhật Adam (vài ms, chưa đo) và không ghi lịch sử loss.

Hệ quả: (a) thay DeepXDE bằng vòng PyTorch thuần cắt tối đa khoảng 15–17% bước; (b) tối ưu đạo hàm bậc hai (sửa đổi 01a, mục 13) nhắm vào 86% còn lại nên là đòn bẩy lớn hơn nhiều; (c) ngưỡng "DeepXDE chiếm > 15% thì xem xét PyTorch thuần" nêu trong thảo luận **không** nằm trong đăng ký, nên không phải quy tắc khóa.

Ghi chú thủ tục: commit `23d94a4` của phiên thứ hai vô tình đưa `fastpde.py` (đang viết dở của phiên chính) vào kho do dùng `git add -A`; nội dung file không do phiên thứ hai viết hay sửa.

### Kết quả 01a/01b (2026-10-06 13:03): E1, E2, E3, E5 đạt; **E4 trượt**

| ID | Kết quả |
|---|---|
| E1 | residual float64 trên 28,859 điểm (anchors, đều, dải giao diện bột/đặc, quanh mặt nghiêng của nêm): lệch 1.80e-13 |
| E2 | gradient theo trọng số (float64): sai tương đối 4.68e-15 |
| E3 | loss PDE float32 trên 364,544 điểm: mới 0.331563 = cũ 0.331563 = S1.2 (lệch 0.0) |
| **E4** | bước cũ **110 ms**, bước mới **159 ms** (**chậm hơn, 0.69×**); bộ nhớ đỉnh 4.6 GB (thấp hơn) |
| E5 | DeepXDE + `fast_residual`: loss bước 0 trùng S1.2; **0.203 s/bước** (so với 0.140 với `pde` cũ) |

Theo quy tắc 01a: **E4 trượt thì giữ `pde` cũ**. Phiên bản đầu của `fastpde` đúng về toán nhưng **không nhanh hơn**: kết quả âm. Dự đoán "E4 đạt (≥ 1.5×)" 70%: sai. Giả thuyết về nguyên nhân (chưa kiểm): `einsum('nwk,vw->nvk')` trên tensor (N, 24, 7) buộc hoán vị bộ nhớ (copy) ở mỗi lớp, trong khi phép nhân ma trận nhỏ 24×24 bị giới hạn bởi băng thông bộ nhớ, không phải tính toán.

### Sửa đổi 01c (2026-10-06 13:04, trước khi chạy): **một** lần thử lại duy nhất với bố cục bộ nhớ khác

Chỉ đổi cách xếp tensor: (K, N, W) với K = 7 kênh (giá trị đạo hàm), thay cho (N, W, K), để phép nhân với W là một GEMM phẳng (K·N, 24) × (24, 24) không cần hoán vị; các phép phần tử giữ nguyên. Toán không đổi, nên E1–E3 phải vẫn đạt. **Tiêu chí E1–E5 giữ nguyên.** Giới hạn: đây là lần thử lại cuối của hướng này; nếu E4 trượt lần nữa thì **bỏ hẳn `fastpde`** (giữ mã trong git làm bản ghi) và không tối ưu bậc hai thêm trong đợt này; hướng khác (nếu có) cần đăng ký riêng. Dự đoán: E4 đạt 45% (mức tăng tốc nếu đạt: 1.5–2.5×).
