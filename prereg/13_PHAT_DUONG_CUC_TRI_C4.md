# Đăng ký trước 13 — phạt dương theo cực trị cho PINN nêm cách nhiệt (C4), để chọn PINN của hồ sơ

Ngày: 2026-10-08 15:45 (đồng hồ). Người dùng đồng ý: (1) chọn PINN của hồ sơ theo quy tắc checkpoint; (2) thử B-min sửa cho C4 (phương án 1 sau khi tôi đính chính rằng đổi `sobol_offset` không nhắm vào cổng (b)); (3) phương án C (bột như đặc, M2'') là dự phòng. **Commit file này trước khi sửa mã và trước khi chạy.** Mọi thay đổi sau đó qua một mục sửa đổi có giờ lấy từ đồng hồ. Làm theo `CLAUDE.md`.

## 1. Bối cảnh và câu hỏi

**Quy tắc chọn PINN của hồ sơ (khóa; dùng cổng huấn luyện đặt trước FD, FD không tham gia chọn):** PINN của hồ sơ cho một hình học là **checkpoint được huấn luyện nhiều nhất mà đạt cổng đầy đủ** (a, b, c của `gates.py`: |E_res| và |E_id| ≤ 10 mJ; hụt T_min ≤ 1% biên độ trên 2²¹ điểm; tỉ lệ năng lượng). Áp dụng không cần chạy thêm: C3 (vết thường) → F08 (`runs/F08/B3s_c`, đạt cổng, hụt 0.72%); C4 (nêm) → F08 trượt cổng (b) (hụt 1.26%) nên **PINN gốc** `runs/Fs/B3W0s_it1` (đạt cổng, hụt 0.84%) **nếu mục này không tạo ra được ứng viên tốt hơn**.

**Chẩn đoán (2026-10-08, không FD, mô tả):** cổng (b) của C4 F08 trượt vì **một điểm** trong 2.1 triệu (T = 231.5 K, tại x = −728, y = −226, z = 222 µm, t/t_end = 1.00; 36 điểm thấp hơn T_amb 20 K; 7208 điểm thấp hơn 5 K). Cổng (b) là thống kê cực trị. Số hạng phạt dương hiện tại (`energy.positivity_term`: trung bình ReLU(−θ)² trên 2¹⁵ điểm đều mỗi bước) hầu như không thấy các điểm đuôi (ước lượng thô đóng góp ~10⁻⁷ vào loss so với loss PDE 0.009). Đổi `sobol_offset` không chạm số hạng này (nó lấy mẫu mới mỗi bước), nên không dùng.

**Câu hỏi:** nếu tiếp tục huấn luyện C4 F08 với số hạng phạt dương nhắm vào cực trị (top-k trên mẫu lớn hơn), mạng có đạt cổng đầy đủ mà không phá phần dư PDE không? Nếu có, ứng viên đó thay PINN gốc làm PINN của C4.

## 2. Thiết kế (khóa)

- **Thay đổi loss duy nhất:** thêm tùy chọn `--pos-topk K` (mặc định 0 = hành vi cũ). Khi K > 0: L_P = trung bình của K giá trị ReLU(−θ)² **lớn nhất** trong **2¹⁷** điểm đều lấy mới mỗi bước (θ = (T − T_amb)/T_c). Mọi số hạng khác không đổi (PDE, biên đỉnh, năng lượng λ_E = 1).
- **Chạy:** `runner.py supervise B3W0s`, khởi động ấm từ `runs/F08/B3W0s_c/B3W0s_98431da4.pt`, envelope `runs/Fs/B3W0s_it0/envelope_next.pt` (như F08), Adam float32, `lr = 1e-4`, `λ_E = 1`, `λ_P = 1000`, **`--pos-topk 16`**, seed 42, `sobol_offset = 0`, **`--adam-iters 10000`**, `--monitor-every 1000`, `--stop-gates 0`, `--time-cap-s 3600`, không L-BFGS. Thư mục ra `runs/F13/B3W0s_c`. Tag khác vì `pos_topk` vào hàm băm.
- **Quy tắc dừng không dùng FD:** hết 10000 bước, hoặc hết 60 phút, hoặc loss không hữu hạn. Một checkpoint (cuối); không chọn "tốt nhất".
- **Trước khi chạy:** kiểm tốc độ (~0.17 s/bước, có thể chậm hơn do mẫu 4 lần lớn hơn) và `nvidia-smi`; một tiến trình GPU một lúc (`lock.py`).

## 3. Đánh giá và quy tắc quyết định (khóa)

**Đủ điều kiện làm PINN của C4 (không dùng FD):** (i) cổng đầy đủ đạt (a, b, c); **và** (ii) hạng phần dư PDE trong loss cuối ≤ 2 × của F08 C4 (0.00874), tức ≤ 0.01748 (bảo vệ phần dư khỏi bị phá). Cả hai thỏa → ứng viên là **PINN của C4**. Một trong hai không thỏa → **PINN của C4 là PINN gốc** `runs/Fs/B3W0s_it1`, và kết quả này được báo cáo là "không đủ điều kiện", không thử lại trong mục này.

**FD:** sau khi quy tắc trên đã áp dụng, đánh giá **một lần** đối chiếu FD cho PINN của C4 được chọn bằng `experiments/verif_table.py` (độ sâu, độ rộng, chiều dài, L2 trường nhiệt) và ΔD (nêm − thường) theo `fd_compare.py` nếu có thể. Chỉ báo cáo, không chấm đạt/không đạt (người dùng đã đồng ý báo cáo sai số đo được).

**Số lần thử:** một lần chạy, một checkpoint. Kết quả nào cũng được báo cáo.

## 4. Dự đoán (ghi trước, có thể sai)

- Đạt cổng (b) sau 10000 bước: ~60%. Đạt (ii): ~75%. Đủ điều kiện (cả hai, cùng cổng a, c): ~45%.
- Nếu đủ điều kiện: độ sâu 70% lệch FD trong ±3% (~75%); L2 vùng nóng trong 13–20% (~70%); lệch nhiệt độ TB vùng nóng vẫn ≤ −100 K (~85%; không kỳ vọng sửa được lệch lạnh).
- Tốc độ ≤ 0.25 s/bước: ~65%; chạy hết 10000 bước trong 60 phút: ~80%.

## 5. Giới hạn (khóa)

Một seed, một hình học (C3 giữ F08 như đã đạt cổng). Số hạng phạt top-k là thay đổi loss, ghi rõ trong luận văn. Quy tắc chọn PINN được viết **sau khi đã thấy** số FD của F08 và PINN gốc (cổng huấn luyện thì có trước FD; việc áp dụng nó làm quy tắc chọn là sau). Không kỳ vọng sửa lệch lạnh vùng sau chùm và lệch ấm vùng lạnh xa chùm; mục này chỉ nhằm cổng (b). Không đổi nghĩa cổng sau khi thấy số.
