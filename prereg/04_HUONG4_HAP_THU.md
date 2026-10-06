# Đăng ký trước 04 — hướng 4: độ nhạy của chỉ báo overhang theo hấp thụ bột α_s

Ngày: 2026-10-06 17:54 (đồng hồ). Người dùng chọn hướng 4 (sau M2″ ĐẠT, đăng ký 03 mục 20). **Commit file này trước khi viết mã và trước khi chạy bất kỳ mốc nào.** Mọi thay đổi sau đó qua một mục sửa đổi có giờ lấy từ đồng hồ. Ngưỡng đánh dấu **[ĐỀ XUẤT]** do tôi đề xuất. Làm theo `CLAUDE.md`.

## 1. Câu hỏi và phạm vi

Hấp thụ hiệu dụng của lớp bột là đầu vào bất định nhất của mô hình (bài Moser 2019: α_s của kim loại đặc dao động 0.3–0.6, và là một trong hai tham số chi phối bất định ở nền). Câu hỏi: **khi α_s đổi trong dải 0.30–0.60, các chỉ báo overhang (độ sâu D, độ xuyên P, tốc độ nguội, T_pk tại đáy lớp bột) của vệt thường và của nêm đổi bao nhiêu, và chỉ báo tương đối (nêm − vệt thường, ΔD) có bền hơn chỉ báo tuyệt đối không?**

Đây là **nghiên cứu độ nhạy**, không phải hiệu chỉnh (M5): không có dữ liệu thực nghiệm trong loss và S0.8 (bảng số thực nghiệm) chưa làm; **không** nói α_s nào đúng. Chỉ trả lời "kết quả phụ thuộc α_s đến đâu". Giữ nguyên các giới hạn của 02 (mục 1, 11): bột dẫn nhiệt như đặc (`powder_k = False`), cận dưới, tương đối, một điều kiện 300 W 0.8 m/s, không hiệu chỉnh.

## 2. α_s đi vào mô hình thế nào (tương quan Moser 2019, Eq. 10, 11; đã dùng sẵn trong `make_scales`)

α_eff = 0.053 + 1.37 α_s − 1.04 α_s² + 0.399 α_s³; β = (0.325 + 1.03 α_s − 1.22 α_s² + 0.587 α_s³)/(D50/2); nguồn laser thể tích Beer-Lambert thay đổi (năng lượng nạp và độ xuyên quang), PDE không đổi. Dẫn nhiệt bột (Eq. 12) **không** phụ thuộc α_s. Số đo tính trước từ `config.scales` (chưa huấn luyện gì):

| α_s | α_eff | β (1/m) | độ xuyên 1/β (µm) | năng lượng nạp (mJ) | T_bal bay hơi (K) |
|---|---|---|---|---|---|
| 0.30 | 0.3812 | 36,003 | 27.8 | 171.5 | 5,100 |
| **0.35 (nền B3)** | 0.4222 | 37,415 | 26.7 | 190.0 | 5,164 |
| 0.45 | 0.4953 | 39,663 | 25.2 | 222.9 | 5,268 |
| 0.60 | 0.5868 | 42,039 | 23.8 | 264.1 | 5,382 |

Năng lượng nạp tăng 54% từ 0.30 đến 0.60; T_bal vẫn dưới T_cap = 6,000 K của mô hình bay hơi. Ngưỡng cổng (10 mJ tuyệt đối, đăng ký 03c) giữ nguyên dù năng lượng nạp đổi.

## 3. Thiết kế

Bảy lượt, mỗi lượt là một lượt M2″ (cùng công thức): bậc không cắt, khởi động từ `B3_84d0289e`, **chỉ Adam** lr 1e-4, loss = L_PDE + L_top + 1·L_E + 1000·L_P, kiểm cổng đầy đủ mỗi 500 bước, dừng khi hai điểm kiểm liên tiếp đạt và đã qua bước 2,000, hoặc hết 6,000 bước, hoặc trần 45 phút; `supervise`; seed như M2″.

| Lượt | Hình học | α_s | Vai trò |
|---|---|---|---|
| **B3r** | vệt thường | 0.35 | **đối chứng công thức**: cùng vật lý với B3 nhưng huấn luyện bằng công thức M2″; sai khác giữa B3r và B3 là **sàn nhiễu của công thức** cho vệt thường |
| B3_a45 | vệt thường | 0.45 | |
| B3W0_a45 | nêm y_a = 0 | 0.45 | |
| B3_a30 | vệt thường | 0.30 | |
| B3W0_a30 | nêm | 0.30 | |
| B3_a60 | vệt thường | 0.60 | |
| B3W0_a60 | nêm | 0.60 | |

Nền α_s = 0.35: vệt thường = B3 (M1, đã qua cổng trung thực: E_id −5.6 mJ), nêm = M2″ (đăng ký 03 mục 20). Mọi lượt khởi động từ B3 (không dùng chuỗi liên tục theo α_s, để các lượt độc lập nhau); nếu một lượt trượt cổng vì khởi động quá xa, báo cáo, và việc dùng chuỗi liên tục chỉ làm qua một sửa đổi riêng. Thứ tự chạy: B3r trước (nếu B3r **trượt** thì dừng chuỗi và báo cáo, vì công thức không hợp lệ cho vệt thường), rồi 0.45, 0.30, 0.60. Lệnh mỗi lượt như M2″ (`python runner.py supervise <bậc> --out runs/H4/<bậc> --adam-iters 6000 --adam-lr 1e-4 --chunk 500 --max-iters 0 --monitor-every 500 --stop-gates 2 --min-steps 2000 --time-cap-s 2700 --energy-lam 1 --pos-lam 1000`). Mã: `config.py` (bậc mới, `RUNS[...]['powder']` ghi đè α_s), `experiments/h4_alpha.py` (chạy chuỗi, bỏ qua lượt đã có `run.json`, tổng hợp). Trần chuỗi 150 phút (dừng ở 225).

**Một lượt ĐẠT khi** đạt tiêu chí M2″: hữu hạn và không sập quá một lần; **dừng vì hai điểm kiểm liên tiếp đạt** và cổng đầy đủ cuối đạt (E_res, E_id ≤ 10 mJ; hụt T_min ≤ 1%; tỉ lệ ≤ 1); `check_physics` 0 [WARN]; ≤ 75 phút. Lượt **trượt cổng** vẫn được ghi, nhưng đánh dấu "không qua cổng" và **không dùng** để kết luận độ nhạy (S1–S3 dưới).

## 4. Mốc test S2.15 (chạy trước chuỗi; trượt thì dừng, báo cáo)

| ID | Kiểm | Tiêu chí |
|---|---|---|
| (a) | Cấu hình: `tag_of` của B3, B3W0, B3W0c, A1s, B1, B2 không đổi (tag cũ); bảy bậc mới có tag đôi một khác nhau và khác B3, B3W0 | đúng |
| (b) | `scales` của mỗi α_s: α_eff, β, năng lượng nạp khớp bảng mục 2 | sai ≤ 1e-3 tương đối |
| (c) | MMS `check_pde` cho nêm ở α_s = 0.30, 0.45, 0.60 | lệch / RMS ≤ 1e-10, không [WARN] |
| (d) | `check_config`, `check_source`, `check_density`, `check_data` cho nêm ở α_s = 0.60 | không [WARN] |
| (e) | Khói `runner.py` cho `B3W0_a60` và `B3_a30` (Adam 40 bước, kiểm 20, `--min-steps 20`, thư mục tạm) | `run.json` DONE, tag đúng bậc, loss hữu hạn |
| (f) | `legacy/ckpt` không đổi (hash đầu và cuối) | đúng |

Trần GPU 8 phút. Dự đoán đạt cả sáu ngay lượt đầu 70%.

## 5. Đầu ra và quy tắc báo cáo

Với mỗi lượt qua cổng, bảng chỉ báo ở 70% (và 50/90%): dài, rộng (W₊ + W₋), sâu D, độ xuyên P = max(D − 60 µm, 0), T_peak; chu trình tại đáy lớp bột (T_pk, thời gian lưu, tốc độ nguội); E_res, E_id, hụt T_min, tỉ lệ. Với mỗi α_s có cả hai trường: ΔD = D_nêm − D_thường, ΔP, và tỉ số tốc độ nguội nêm/thường, ΔT_pk.

**Sàn nhiễu** (đo, không giả định): n_D = max(|D_thường(B3r) − D_thường(B3)|, 1.2 µm); 1.2 µm là khoảng dao động độ sâu 70% giữa bảy biến thể huấn luyện trên nêm (đăng ký 03, mục 17, 20).

**Các phát biểu kiểm tra được (khóa trước khi chạy), dùng các lượt qua cổng, cặp α_s xa nhất có đủ dữ liệu:**
- **S1 (độ sâu phụ thuộc α_s, vượt sàn nhiễu):** |D(α_max) − D(α_min)| ≥ 2 · n_D cho cả vệt thường và nêm. Dự đoán đúng 90%.
- **S2 (chỉ báo tương đối bền hơn tuyệt đối):** khoảng biến thiên của ΔD theo α_s ≤ 0.5 × khoảng biến thiên của D_nêm. Dự đoán đúng 40% (nếu hiệu ứng nêm tỉ lệ với độ sâu, ΔD cũng đổi theo α_s).
- **S3 (hiệu ứng nêm còn tách khỏi sàn nhiễu ở mọi α_s qua cổng):** ΔD(α) ≥ 2 · n_D ở mọi α_s qua cổng (ΔD(0.35) = +9.0 µm ở M2″). Dự đoán đúng 85%.
- **Chỉ ghi:** tỉ số ΔD/D_thường theo α_s; độ nhạy tuyến tính (dD/dα_s, dΔD/dα_s) với hai đầu mút; tốc độ nguội và T_pk. Nếu một đầu mút trượt cổng, S1–S2 dùng cặp xa nhất còn lại và ghi rõ.

## 6. Dự đoán chung (ghi trước, có thể sai)

- Từng lượt qua cổng: 70% (thấp hơn ở α_s = 0.60: 55%, vì khởi động xa nền B3 và năng lượng nạp +39%).
- Cả bảy lượt qua cổng: 20%; B3r và cả cặp (0.45) qua cổng: 55%; ít nhất một cặp (vệt thường, nêm) ngoài α_s = 0.35 qua cổng: 80%.
- D tăng đơn điệu theo α_s ở vệt thường: 95%. Khoảng của D_thường giữa α_s = 0.30 và 0.60: khoảng 15–35 µm (ước lượng thô: D tăng gần tỉ lệ năng lượng nạp^0.7).
- Tổng GPU: 60–110 phút (mỗi lượt 6–16 phút); tối đa 150.

## 7. Giới hạn (khóa)

Bất định đầu vào chỉ gồm α_s (một tham số quét bốn điểm, không phải phân bố xác suất đầy đủ): không gồm sai số ±0.02 và ±11% của các tương quan Moser, k bột (đang bằng k đặc), độ xốp, D50, hay hiệu ứng nhiều lớp. Dải 0.30–0.60 là của thép 316L trong bài Moser, không phải đo cho IN718 ở dự án này; ở α_s lớn (≥ 0.45) chế độ thực của quá trình có thể chuyển sang keyhole mà mô hình (không có áp suất giật, đối lưu chỉ xấp xỉ bằng k tăng) không biểu diễn. Kết quả là độ nhạy **trong mô hình này**, không thay thế hiệu chỉnh (M5). Mỗi lượt là một lần huấn luyện (chưa có M3, độ ổn định theo tập điểm) nên sàn nhiễu n_D là ước lượng từ các biến thể đã có, không phải độ lệch chuẩn đầy đủ.
