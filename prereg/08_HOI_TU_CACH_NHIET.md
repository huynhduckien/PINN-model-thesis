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
