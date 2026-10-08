# Đăng ký trước 12 — tinh chỉnh từ mạng khớp bằng chính loss PINN: tối ưu hay dạng loss?

Ngày: 2026-10-08 13:42 (đồng hồ). Người dùng đồng ý đề xuất này sau `prereg/11`. **Commit file này trước khi viết mã và trước khi chạy bất kỳ lượt nào.** Mọi thay đổi sau đó qua một mục sửa đổi có giờ lấy từ đồng hồ. Ngưỡng đánh dấu **[QUY ƯỚC]** là lựa chọn của tôi, người dùng quyết. Làm theo `CLAUDE.md`.

## 1. Câu hỏi và phạm vi

`prereg/11` cho thấy loss PDE của PINN F08 (0.005–0.009) thấp hơn hơn 400 lần so với loss của mạng khớp FD (4.0–4.2), dù mạng khớp chính xác hơn nhiều về giá trị (RMS 11–13 K so với 135–206 K). Phép đo đó bị nhiễu đạo hàm của mạng khớp nên **không phân biệt được** hai giả thuyết:
- **(T) Tối ưu:** tồn tại nghiệm chính xác về giá trị VÀ loss PDE thấp trong lớp hàm của mạng; PINN F08 chưa tìm được (bị kẹt).
- **(L) Dạng loss:** các nghiệm chính xác về giá trị có loss PDE cao hơn các nghiệm sai trong lớp hàm; khi giảm loss, trường trôi về nghiệm sai.

**Câu hỏi:** nếu khởi động từ mạng khớp (chính xác về giá trị) rồi tinh chỉnh **chỉ bằng loss PINN** (phần dư PDE, biên, năng lượng, dương; không dữ liệu), thì khi loss PDE giảm về mức của PINN F08, trường nhiệt giữ gần FD (T) hay trôi về nghiệm sai của PINN (L)?

**Đây là phép chẩn đoán.** Khởi động từ mạng khớp dùng dữ liệu FD ở khâu khởi tạo, nên **không** cho ra PINN thuần vật lý của hồ sơ và không đổi quy tắc "không FD, không dữ liệu trong loss" cho PINN. Kết quả chỉ chỉ hướng: (T) → đổi tối ưu; (L) → đổi hàm loss.

## 2. Thiết kế (khóa)

- **Hai ca** C3 (`B3s`, bao `runs/ENV/B3_iter0.pt`) và C4 (`B3W0s`, bao `runs/Fs/B3W0s_it0/envelope_next.pt`), khởi tạo từ trọng số mạng khớp `runs/FD/capacity_C3_A_r2.pt`, `capacity_C4_A_r2.pt` (kiến trúc 6×24 như PINN).
- **Huấn luyện:** cùng PDE, mặt nạ, loss và siêu tham số như F08: Adam float32, `lr = 1e-4`, `λ_E = 1`, `λ_P = 1000`, seed 42, điểm collocation mặc định (`config.SAMPLING`), mã DeepXDE qua `runner.build`; trạng thái Adam liên tục giữa các đoạn. Không dừng theo cổng, không dùng FD để quyết định gì khi huấn luyện.
- **Đoạn và dừng:** huấn luyện theo **đoạn 2500 bước**; sau mỗi đoạn ghi trọng số, loss bốn thành phần (từ `lh.loss_train`), và sai số so với FD (mục 3). **Dừng** ở đoạn đầu tiên mà loss PDE ≤ 2 × loss PDE của PINN F08 (C3 ≤ 0.01090; C4 ≤ 0.01748) **[QUY ƯỚC: 2×]**, hoặc sau **20000 bước** (8 đoạn) nếu chưa đạt, hoặc nếu loss không hữu hạn. Quyết định dừng chỉ dùng loss, không dùng sai số FD.
- **Thời gian:** ước tính 0.17 s/bước (phép đo 08), tối đa ~57 phút mỗi ca. Kiểm `nvidia-smi` trước khi chạy; GPU một tiến trình; ghi lệnh vào log.
- **FD để đo sai số:** FD 10 µm của `fdv2` (như `fd_errormap.py`) tại 50/70/90% quét, tính một lần mỗi ca.

## 3. Đo và phân loại (khóa)

**Đo mỗi đoạn:** loss PDE; RMS nóng (căn trung bình bình phương (T_mạng − T_FD) trên ô T_FD > 600 K, tại 50/70/90% quét); lệch trung bình vùng sau chùm và vùng quanh chùm tại 70% (định nghĩa vùng như `fd_errormap.py`).

**Mốc PINN F08 (70% quét, `prereg/10`):** RMS nóng C3 135.3 K, C4 205.8 K; lệch vùng sau chùm C3 −152.3 K, C4 −232.4 K.

**Điểm đánh giá:** đoạn dừng (mục 2).

**Phân loại cho từng ca:**
- **TỐI ƯU (T):** loss PDE ≤ 2 × loss PINN F08 **và** RMS nóng (70%) ≤ 50% của PINN F08 (C3 ≤ 67.6 K; C4 ≤ 102.9 K) **và** \|lệch vùng sau chùm\| ≤ 50% của PINN F08 (C3 ≤ 76 K; C4 ≤ 116 K) **[QUY ƯỚC: 50%]**.
- **DẠNG LOSS (L):** loss PDE ≤ 2 × loss PINN F08 **và** RMS nóng (70%) ≥ 75% của PINN F08 (C3 ≥ 101 K; C4 ≥ 154 K) **[QUY ƯỚC: 75%]**.
- **KHÔNG KẾT LUẬN:** loss PDE chưa đạt ≤ 2 × loss PINN F08 trong 20000 bước (khi đó chỉ báo quỹ đạo, ghi rõ), hoặc nằm giữa hai nhóm trên.

**Phân loại chung:** (T) nếu cả hai ca (T); (L) nếu cả hai ca (L); còn lại KHÔNG KẾT LUẬN / HỖN HỢP.

**Báo cáo kèm:** quỹ đạo (loss, RMS nóng, lệch vùng sau chùm) theo đoạn; phần dư tương đối gần chùm của trọng số cuối (cách `fd_fieldcheck.residual_stats`); loss ba thành phần còn lại.

## 4. Dự đoán (ghi trước, có thể sai)

- Loss PDE đạt ≤ 2 × loss PINN F08 trong 20000 bước: **~40%** (từ 4.2 xuống ~0.01: giảm 400 lần; PINN F08 tự cần 20000 bước để đi từ 0.0097 xuống 0.0055).
- Nếu đạt: (T) ~40%, (L) ~40%, hỗn hợp ~20%. Tổng thể: (T) ~16%, (L) ~16%, KHÔNG KẾT LUẬN / HỖN HỢP ~68%.
- RMS nóng (70%) tăng so với mạng khớp ban đầu (10.9 / 12.6 K) ngay sau đoạn đầu (loss PDE kéo trường đi): ~75%.
- Sai số giá trị không đơn điệu theo loss (có giai đoạn tăng rồi giảm): ~30%.
- Tổng thời gian dưới 1.5 giờ: ~55%.

## 5. Giới hạn (khóa)

Một seed, một lần chạy mỗi ca; lấy mẫu mặc định (không nhánh W); "loss ≤ 2× loss PINN" là mốc quy ước; điểm đánh giá là điểm dừng, không phải cực trị; khởi tạo từ mạng khớp là đường đi hẹp trong không gian tham số nên (T) chỉ chứng minh tồn tại cực trị tốt gần mạng khớp chứ không chứng minh PINN từ khởi tạo ngẫu nhiên tìm được nó; (L) chứng minh hàm loss kéo trường ra khỏi vùng chính xác **theo hướng gradient của Adam ở lr này**, không chứng minh mọi cực trị của loss đều sai. Không đổi bảng chỉ báo, không phải PINN của hồ sơ.


---

## Sửa đổi A1 (2026-10-08 14:30, đồng hồ): dừng sớm C3 ở bước 12500 và chạy C4 ngắn (5000 bước); viết sau khi đã thấy quỹ đạo C3

**Công khai về thời điểm.** Mục này đổi điều kiện dừng đã khóa ở mục 2 **sau khi đã thấy quỹ đạo C3 đến bước 12500**, theo quyết định của người dùng (dừng và chạy C4 ngắn để kiểm hiện tượng có lặp lại ở hình học nêm). Lý do: loss PDE giảm quá chậm (0.233 → 0.144 → 0.122 → 0.113 → 0.108) so với mức dừng 0.0109, nên nhãn theo mục 3 gần như chắc chắn là KHÔNG KẾT LUẬN (loss chưa đạt mức PINN) dù chạy hết 20000 bước; chạy hết tốn thêm ~1.6–2.3 giờ mà ít thông tin thêm.

**Quỹ đạo C3 (log `runs/FD/finetune.log`; trọng số các đoạn `runs/FD/ft_C3_seg1…5.pt`; tiến trình bị dừng ở đoạn 6, không có `finetune_C3.json`):**

| Bước | Loss [PDE, biên, năng lượng, dương] | RMS nóng 70% (K) | Lệch vùng sau chùm | Lệch quanh chùm |
|---|---|---|---|---|
| 0 (mạng khớp) | [4.222, 0.494, 0.0198, 0.0021] | 10.9 | −1.1 | +0.5 |
| 2500 | [0.2334, 0.0361, 0.0494, 0.0072] | 374.0 | −284.8 | −137.2 |
| 5000 | [0.1443, 0.0285, 0.0073, 0.0074] | 393.4 | −333.9 | −70.9 |
| 7500 | [0.1221, 0.0242, 0.0088, 0.0061] | 432.7 | −403.1 | −41.0 |
| 10000 | [0.1125, 0.0209, 0.0103, 0.0045] | 457.0 | −440.6 | −11.9 |
| 12500 | [0.1076, 0.0176, 0.0032, 0.0067] | 462.4 | −442.7 | +22.6 |

Mốc PINN F08 (70%): loss PDE 0.00545, RMS nóng 135.3 K, lệch sau chùm −152.3 K. **Nhãn C3 (theo mục 3): KHÔNG KẾT LUẬN (loss chưa đạt mức PINN)**, chưa đủ để chọn (T) hay (L).

**C4 ngắn (khóa trước khi chạy):** cùng thiết kế mục 2 (cùng siêu tham số, khởi tạo từ `runs/FD/capacity_C4_A_r2.pt`), nhưng **chỉ 2 đoạn = 5000 bước** (`finetune_from_fit.py --cases C4 --max-seg 2`). Dừng sớm hơn nếu loss PDE ≤ 2 × loss PINN F08 (≤ 0.01748) hoặc loss không hữu hạn. Tiêu chí "hiện tượng lặp lại ở nêm" (khóa, đối chiếu quỹ đạo C3 ở bước 5000: RMS nóng 393 K = 2.9 lần mốc PINN F08): **LẶP LẠI** nếu RMS nóng (70%) của C4 ở bước 5000 ≥ 1.5 × mốc PINN F08 của C4 (≥ 309 K) **và** lệch vùng sau chùm ≤ −116 K; **KHÔNG LẶP LẠI** nếu RMS nóng ≤ mốc PINN F08 (≤ 205.8 K); **KHÔNG RÕ** còn lại **[QUY ƯỚC: 1.5×, −116 K]**. Phân loại (T)/(L) theo mục 3 chỉ áp dụng nếu loss đạt mức PINN (khó xảy ra ở 5000 bước).

**Dự đoán C4 (ghi trước):** LẶP LẠI ~65%; KHÔNG RÕ ~25%; KHÔNG LẶP LẠI ~10%. Loss PDE ở bước 5000 vẫn > 0.0175 (chưa đạt mức dừng): ~90%. Tổng thời gian dưới 30 phút: ~60%.

**Giới hạn:** hai ca với độ dài khác nhau (C3 12500 bước, C4 5000 bước) nên không so được ở bước lớn; một seed; kết luận (T)/(L) vẫn không thể có từ 12; kết quả chỉ nói về xu hướng của đường đi giảm loss từ mạng khớp ở lr 1e-4.
