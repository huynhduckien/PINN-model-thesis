# Đăng ký trước 08 — huấn luyện PINN cách nhiệt đến hội tụ (bước 1 sau `prereg/07`)

Ngày: 2026-10-08 00:16 (đồng hồ). Người dùng đồng ý: làm bước 1 trước (huấn luyện lại hai PINN cách nhiệt đến hội tụ, không dừng ở "cổng đạt"; lấy FD tham chiếu làm tiêu chí chấp nhận). **Commit file này trước khi sửa mã và trước khi chạy bất kỳ lượt nào.** Mọi thay đổi sau đó qua một mục sửa đổi có giờ lấy từ đồng hồ. Ngưỡng đánh dấu **[ĐỀ XUẤT]** do tôi đề xuất, người dùng quyết. Làm theo `CLAUDE.md`.

## 1. Câu hỏi và giả thuyết

`prereg/07` kết luận PINN cách nhiệt **SAI** ở ΔD (+18.3 so +26.8 µm của FD) và ΔT_pk (+160 so +180 K). Bằng chứng chẩn đoán cho thấy lượt `B3W0s_it1` dừng sau **2000 bước Adam, 347 giây, không L-BFGS**, vì quy tắc `stop_gates` ("cổng đạt") khi hạng phần dư PDE trong loss còn 1.4×10⁻² (gấp ~10 lần B3), phần dư PDE trung vị 17–19% so với các số hạng, và phần dư theo dải nhiệt độ bù trừ nhau (−2.7, +32.7, −19.7, −8.5 mJ cộng lại +1.8 mJ).

**Giả thuyết H1:** sai số chủ yếu do **chưa hội tụ** (huấn luyện dừng sớm). **Câu hỏi:** nếu huấn luyện tiếp hai PINN cách nhiệt (C3 `B3s`, C4 `B3W0s`) bằng cùng PDE, cùng mặt nạ, cùng điểm lấy mẫu, cùng loss, nhưng **không dừng theo cổng** và với ngân sách lớn hơn, thì các Δ của bảng chỉ báo có vào ngân sách của `prereg/07` không?

**Không phải câu hỏi này:** PDE có đúng vật lý không; các giả thuyết H2 (sức chứa mạng) và H3 (lấy mẫu) không nằm trong 08. Nếu H1 không đủ, các giả thuyết đó cần đăng ký riêng (09).

## 2. Thiết kế (khóa)

- **Hai lượt, tiếp tục từ checkpoint hiện có** (khởi động ấm, trạng thái Adam khởi tạo lại), giữ đúng PDE và mặt nạ mà FD đã dùng:

| Ca | Bậc (`config.RUNS`) | Khởi tạo từ | Bao T_max cố định |
|---|---|---|---|
| C3 | `B3s` | `runs/Fs/B3s_it0/B3s_d2f9f1db.pt` | `runs/ENV/B3_iter0.pt` (sha 20034cb6) |
| C4 | `B3W0s` | `runs/Fs/B3W0s_it1/B3W0s_ba6419f6.pt` | `runs/Fs/B3W0s_it0/envelope_next.pt` (sha 71d24a28) |

  C1, C2 (bột như đặc) **không** huấn luyện lại; mặt nạ **không** lặp lại (không có vòng `f_insul`).
- **Huấn luyện:** Adam float32, `lr = 1e-4`, `λ_E = 1`, `λ_P = 1000` (như `B3W0s_it1`), seed 42, `sobol_offset = 0` (cùng tập điểm), `--monitor-every 1000`, **`--stop-gates 0`** (không dừng theo cổng), `--adam-iters` **[ĐỀ XUẤT] 20000**, `--time-cap-s` **[ĐỀ XUẤT] 3600** (60 phút mỗi lượt). Không L-BFGS (`--max-iters 0`): runner không hỗ trợ L-BFGS với số hạng phụ ngẫu nhiên (`λ_E`, `λ_P`); đưa L-BFGS vào cần thiết kế mới, để dành cho 09.
- **Quy tắc dừng không dùng FD:** lượt dừng khi hết 20000 bước Adam, hoặc khi hết 60 phút, hoặc khi loss không hữu hạn. FD **không** được dùng để quyết định dừng, chọn checkpoint hay đổi siêu tham số (tránh chỉnh PINN theo bộ kiểm). Mỗi lượt cho **một** checkpoint (cuối), không chọn "tốt nhất" theo FD.
- **Trước khi chạy dài:** kiểm tốc độ (~0.17 s/bước) và `nvidia-smi`; GPU một tiến trình một lúc (`lock.py`); chạy lần lượt C3 rồi C4. Các lượt chạy qua `runner.py supervise` (như `experiments/f_insul.py`), thư mục ra `runs/F08/`.

## 3. Đánh giá (khóa)

Một lần duy nhất sau khi cả hai lượt xong, đúng giao thức `prereg/07`: tính chu trình nhiệt PINN mới bằng `experiments/pinn_cycles.py` (cho C3, C4 mới), rồi `experiments/fd_compare.py compare` với FD **cố định** (`runs/FD/fd_cases.json`, lưới 5 µm, phán quyết hai cận p = 2 và p = 1, giống A5). **Ngân sách giữ nguyên số của `prereg/07`** (tol = 10% hiệu ứng của PINN cũ: 1.83 µm, 15.98 K, 22.2 µs; `EFFECT` không đổi để mục tiêu không bị dời).

**Thông tin chấp nhận (khóa):**
- **ĐẠT** nếu ΔD, ΔT_pk, Δ thời gian lưu cùng ĐÚNG (hai cận). Tỉ số nguội vẫn KHÔNG XÁC ĐỊNH (FD; cần đổi điểm đo, việc riêng) và **không** chặn.
- **CẢI THIỆN** (chưa đạt): \|lệch\| của ΔD, ΔT_pk giảm ít nhất một nửa so với `prereg/07` (8.46 µm, 20.1 K) nhưng chưa ĐÚNG.
- **KHÔNG ĐỔI / XẤU HƠN:** còn lại.

**Chẩn đoán (chỉ báo cáo, không chấm):** độ sâu, độ rộng, chiều dài, nhiệt độ đỉnh của C3, C4 so FD; trung bình và RMS (T_PINN − T_FD) theo vùng (`fd_errormap.py`) và dọc đường (`fd_fieldcheck.py`); năng lượng lưu trữ và E_id; phần dư PDE chuẩn hóa (trung vị, phân vị 90); đường cong loss ở các điểm kiểm (1000 bước một); **[ĐỀ XUẤT]** cổng nghiêm hơn chỉ để báo cáo: |E_id| ≤ 2 mJ và phần dư trung vị ≤ 5%.

**Số lần thử:** mục này cho phép **một** lần đánh giá đối chiếu FD. Nếu kết quả là CẢI THIỆN hoặc KHÔNG ĐỔI, bước tiếp theo (H2, H3, L-BFGS, đổi sampling) cần đăng ký 09 trước; không tinh chỉnh tiếp trong 08 sau khi thấy số FD.

## 4. Dự đoán (ghi trước, có thể sai)

- Loss (hạng phần dư PDE) giảm rõ rệt so với 1.4×10⁻² nhưng chưa về mức B3: giảm ít nhất 2 lần (~60%), ít nhất 5 lần (~25%).
- Độ lệch trung bình (T_PINN − T_FD) trên vùng T > 1000 K của C3, C4 giảm ít nhất một nửa (~40%).
- C4: ΔD chuyển sang CẢI THIỆN hoặc ĐẠT (~45%); ĐẠT (cả ba Δ ĐÚNG) ~15%. Lý do bất định: H2 (sức chứa mạng 6×24) và H3 (lấy mẫu vùng sâu/bột dưới nêm) chưa kiểm; lỗi hiện tại cũng lớn trong vùng được lấy mẫu dày nên chỉ huấn luyện thêm có thể không đủ.
- Năng lượng lưu trữ của PINN tiến gần FD (thiếu hụt còn dưới 3%) ~35%.
- Một lượt có thể không dùng hết 20000 bước trong 60 phút (nếu tốc độ chậm hơn ~0.18 s/bước): ~45%.

## 5. Giới hạn (khóa)

Một điều kiện, một hình học, mặt nạ cố định (đúng FD), FD hai lưới (10 và 5 µm) với giai đoạn B bỏ (A5); chỉ trả lời việc PINN cách nhiệt có giải đúng **PDE hiện tại** hay không, không nói về vật lý. Kết quả CẢI THIỆN hoặc ĐẠT trên một seed chưa nói về độ lặp lại; muốn khẳng định cần thêm seed (đăng ký riêng). Dùng FD làm tiêu chí chấp nhận là kiểm **một lần** theo mục 3, không phải vòng chỉnh theo FD.


---

## Kết quả 08 (2026-10-08 02:16, đồng hồ): ΔD vào ngân sách, nhưng Δ lưu xấu đi và ΔT_pk gần như không đổi; nhãn theo mục 3 là KHÔNG ĐỔI / XẤU HƠN

**Huấn luyện (theo mục 2, không FD trong quyết định nào):** cả hai lượt chạy hết 20000 bước Adam (C3 `B3s_92803317`: 3355 s; C4 `B3W0s_98431da4`: 3395 s, dưới trần 3600 s), dừng vì ngân sách bước. Hạng phần dư PDE của loss: C3 0.00973 → 0.00545 (giảm 1.78 lần), C4 0.01438 → 0.00874 (giảm 1.65 lần). Cổng đầy đủ: C3 **đạt** (E_id −7.0 mJ, hụt T_min 0.72%); C4 **trượt cổng (b)**: T_min 231.5 K, hụt dưới T_amb 66.5 K = 1.26% biên độ (> 1%; lượt cũ 0.84%), `check_physics` 1 cảnh báo; E_res +3.45 mJ, E_id −2.33 mJ.

**Đánh giá chính thức, một lần (`fd_compare.py compare --tag f08`, `runs/FD/compare_f08.json`; FD cố định, lưới 5 µm, hai cận p = 2 và p = 1):**

| Δ (nêm − vệt thường) | PINN cũ (07) | PINN mới | FD | \|lệch\| mới (cũ) | tol | Phán quyết mới (cũ) |
|---|---|---|---|---|---|---|
| ΔD cách nhiệt (µm) | +18.31 | **+26.01** | +26.77 | **0.77** (8.46) | 1.83 | **ĐÚNG** (SAI) |
| ΔT_pk (K) | +159.8 | +161.4 | +179.9 | 18.5 (20.1) | 15.98 | KHÔNG KẾT LUẬN (SAI): p = 2 SAI, p = 1 không kết luận |
| Δ thời gian lưu (µs) | +222.0 | **+307.0** | +230.3 | **76.7** (8.3) | 22.2 | **SAI** (ĐÚNG) |
| tỉ số nguội | 0.5425 | 0.549 | không xác định | – | 0.0457 | KHÔNG XÁC ĐỊNH |
| ΔD bột như đặc | +8.99 | +8.99 (không huấn luyện lại) | +9.62 | 0.63 | 0.90 | KHÔNG KẾT LUẬN |

**Phán quyết bảng chỉ báo (C4 − C3): SAI** (Δ lưu SAI). **Theo mục 3, nhãn là KHÔNG ĐỔI / XẤU HƠN, không phải CẢI THIỆN:** ΔD đã giảm lệch hơn một nửa (8.46 → 0.77 µm) nhưng ΔT_pk không giảm một nửa (20.1 → 18.5 K) và Δ lưu xấu đi (ĐÚNG → SAI). Điều kiện ĐẠT (cả ba Δ ĐÚNG) không thỏa. Nhãn này giữ theo định nghĩa đã khóa, dù ΔD đã khớp.

**Lý do Δ lưu xấu đi (từ số, không phải đoán):** `prereg/07` ghi Δ lưu ĐÚNG (lệch 8.3 µs), nhưng thời gian lưu tuyệt đối của PINN cũ thấp hơn FD ở cả hai ca (C3: 432.5 so 550.1 µs, −118; C4: 654.5 so 780.5 µs, −126) và hai sai số **triệt tiêu nhau** trong hiệu. PINN mới ở C4 tiến gần FD (753.0 so 780.5 µs, −27) nhưng ở C3 gần như không đổi (446.0 so 550.1, −104), nên hiệu lệch thành +307 so +230. Phán quyết ĐÚNG của Δ lưu ở 07 là do bù trừ sai số, không phải do PINN đúng. Ngược lại ΔD khớp thật: độ sâu từng ca đều lệch FD khoảng 1 µm (C3 90.4 so 90.8; C4 116.4 so 117.6).

**Chẩn đoán (chỉ báo cáo; FD 10 µm, `fieldcheck_f08.json`, `errormap_f08.json`; PINN cũ trong ngoặc):**
- Độ sâu / rộng / dài (µm, tự ghi): C3 90.4 / 186.5 / 514 (89.5 / 174.3 / 503; FD 90.8 / 194.3 / 679); C4 116.4 / 193.8 / 587 (107.8 / 173.4 / 514; FD 117.6 / 204.8 / 773).
- Trung bình (T_PINN − T_FD) vùng tập trung quanh chùm: C3 −80 K (−140), C4 −86 K (−174). Vùng sau chùm (ngoài 16 r_b): C3 −152 K (−130), C4 −232 K (−262): gần như không đổi. Dọc đường (trung bình): C3 dọc −58 (−96), ngang −83 (−118), dọc x −114 (−139); C4 dọc −58 (−122), ngang −84 (−130), dọc x −123 (−197).
- T tại t = 1500 µs ở điểm chu trình 50%: C3 1153 K (1186; FD 1350), C4 1513 K (1406; FD 1640).
- Phần dư PDE của PINN chuẩn hóa (trung vị / phân vị 90): C3 14.1% / 43% (17.5% / 64%), C4 18.4% / 68.5% (19.2% / 77%).
- Năng lượng lưu trữ so FD (cổng đầy đủ, 5 µm): C3 123.6 so 138.9 mJ (−11.0%; cũ −11.7%), C4 143.8 so 149.3 mJ (−3.7%; cũ −6.4%).
- Cổng nghiêm hơn chỉ để báo cáo ([ĐỀ XUẤT]): |E_id| ≤ 2 mJ: C3 không (−7.0), C4 không (−2.3, gần); phần dư trung vị ≤ 5%: cả hai không.

**Chấm dự đoán (mục 4):**

| Dự đoán | Kết quả |
|---|---|
| hạng phần dư PDE giảm ≥ 2 lần (~60%), ≥ 5 lần (~25%) | **sai**: 1.65–1.78 lần |
| lệch trung bình T giảm ≥ một nửa ở C3, C4 (~40%) | **chưa đạt cho cả hai**: vùng tập trung giảm 43% (C3), 51% (C4); các đường dọc / ngang / dọc x giảm 18–52% (chỉ đường dọc của C4 đạt một nửa, 52%); vùng sau chùm không giảm (C3 xấu hơn 17%: −130 → −152 K; C4 giảm 11%) |
| C4: ΔD CẢI THIỆN hoặc ĐẠT (~45%); ĐẠT cả ba Δ (~15%) | ΔD **ĐÚNG** (đúng dự đoán); cả ba Δ ĐÚNG: **không** |
| năng lượng lưu trữ thiếu hụt còn dưới 3% (~35%) | **sai**: C3 −11.0%, C4 −3.7% |
| một lượt có thể không dùng hết 20000 bước (~45%) | **sai**: cả hai dùng hết, 3355 và 3395 s |

**Kết luận về H1:** huấn luyện thêm (không dừng theo cổng) **sửa được độ sâu và ΔD**, và kéo gần hơn độ rộng, chiều dài, nhiệt độ ở vùng tập trung; **không sửa được vùng đuôi** (−150 đến −230 K), thiếu hụt năng lượng của C3, thời gian lưu tuyệt đối của C3 và ΔT_pk. Phần dư PDE chỉ giảm 1.65–1.78 lần, nên mô hình vẫn còn xa mức hội tụ của B3. H1 đúng một phần; phần còn lại cần đăng ký 09 (các giả thuyết chưa kiểm: sức chứa mạng, lấy mẫu vùng sau chùm và vùng bột, L-BFGS cần thiết kế mới vì số hạng phụ ngẫu nhiên).

**Giới hạn:** một seed, một lần đánh giá theo mục 3; không chỉnh gì theo FD sau khi thấy số. C4 mới trượt cổng (b) nên không đủ điều kiện làm "PINN của hồ sơ" nếu dự án áp dụng cổng đầy đủ; các số trên chưa lan sang W = 45 µm hay M3 (chưa huấn luyện lại).
