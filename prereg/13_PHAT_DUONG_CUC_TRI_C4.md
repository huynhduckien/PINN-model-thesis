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

---

## Sửa đổi A1 (2026-10-08 15:46, đồng hồ): quy tắc dừng sớm theo monitor; viết sau khi đã khởi chạy (15:42) và **trước khi đọc bất kỳ dòng monitor nào**

**Công khai về thời điểm.** Mục này thêm quy tắc dừng sớm vào mục 2, theo đề nghị của người dùng ("nếu quá sai mà không thấy cải thiện thì dừng trước 30 phút"). Lượt chạy đã bắt đầu lúc 15:42 với quy tắc cũ; chưa có dòng monitor nào được đọc khi viết mục này (log chỉ có dòng khởi động). Không dùng FD và không dùng cổng đầy đủ để dừng.

**Số đo dùng để dừng** (lấy từ `last.pt`, `adam.monitor[*].loss_raw`, sau mỗi 1000 bước; `loss_raw = [PDE, biên, λ_E·L_E, λ_P·L_P]`). Chú ý: cổng nhẹ trong log chỉ có 2¹⁸ điểm nên không thấy được thống kê cực trị của cổng (b); vì vậy dùng chính số hạng phạt nhắm cực trị làm tín hiệu.

**Quy tắc (khóa, kiểm tại bước 3000 và bước 5000; dừng ở lần đầu một điều kiện thỏa):**
- **D1 (phá phần dư PDE):** `loss_raw[0]` > 0.01748 (2 × F08 C4) tại bước 3000 hoặc 5000 → dừng.
- **D2 (không cải thiện số hạng nhắm cực trị):** tại bước 5000, `loss_raw[3]` ≥ 0.5 × `loss_raw[3]` tại bước 1000 → dừng.
- Đồng hồ: bước 5000 xảy ra ở ~14–21 phút nếu tốc độ 0.17–0.25 s/bước; nếu tốc độ chậm hơn và bước 5000 chưa tới sau 30 phút kể từ 15:42 (tức 16:12) thì dừng ở đó với nhãn "chậm".
- Dừng sớm = kết thúc tiến trình, không có checkpoint hoàn chỉnh, nên **không đủ điều kiện** (mục 3); PINN của C4 là PINN gốc theo quy tắc mục 1. Không thử lại trong mục này.
- Qua được cả hai điểm kiểm thì chạy tiếp đến 10000 bước như mục 2.

**Giới hạn:** các ngưỡng 0.5× và 0.01748 là **[QUY ƯỚC]** của tôi, đặt trước khi thấy monitor; D2 chỉ đo xu hướng của số hạng phạt chứ không chứng minh cổng (b) sẽ đạt.

---

## Sửa đổi A2 (2026-10-08 16:2x, đồng hồ): D2 kích hoạt ở bước 5000; người dùng chọn ngoại lệ — xét checkpoint bước 5000 như ứng viên hoàn chỉnh; viết **trước** khi chạy cổng đầy đủ (a, c) và FD

**Sự kiện (công khai).** Bước 5000 tới lúc 16:10:31 (trước mốc "chậm" 16:12). D1 không kích hoạt (PDE 0.008666 ≤ 0.01748). **D2 kích hoạt** (λ_P·L_P 7.444×10⁻⁶ ≥ 0.5 × 2.993×10⁻⁶ ở bước 1000). Tiến trình bị dừng theo A1; `runs/F13/B3W0s_c/last.pt` (bước 5000) còn nguyên. **Sau khi D2 kích hoạt** tôi đã đo cổng (b) trên 2²¹ điểm (seed 101, chỉ đọc, không FD) cho checkpoint này: T_min 296.6 K, hụt 0.03%, không điểm nào thấp hơn T_amb − 5 K (C4 F08: 231.5 K, hụt 1.26%, 7208 điểm). Tôi đã thấy số này **trước** khi viết A2.

**Lỗi thiết kế của D2 (thừa nhận).** Mốc so sánh là một lần lấy mẫu ở bước 1000, vốn đã rất thấp (3×10⁻⁶) và dao động mạnh giữa các bước (3×10⁻⁶ → 5.4×10⁻⁵ → 1.5×10⁻⁵ → 8.4×10⁻⁶ → 7.4×10⁻⁶), nên D2 gần như chắc chắn kích hoạt dù mục tiêu đã đạt.

**Ngoại lệ (người dùng chọn):** checkpoint bước 5000 được xét như một ứng viên hoàn chỉnh. **Quy tắc đủ điều kiện của mục 3 giữ nguyên, không đổi:** (i) cổng đầy đủ đạt (a, b, c; b đã đo ở trên); (ii) hạng phần dư PDE trong loss ≤ 0.01748 (đã có: 0.008666). Cổng (a) và (c) chưa đo đầy đủ; đo bằng `runner.full_gate` (E_res, E_id, tỉ lệ năng lượng). Đủ điều kiện → PINN của C4, đánh giá FD **một lần** (mục 3). Không đủ điều kiện → PINN của C4 là PINN gốc `runs/Fs/B3W0s_it1`. Không huấn luyện thêm; không đổi ngưỡng nào.

**Giới hạn:** ngoại lệ này được chọn sau khi thấy số cổng (b); lựa chọn dựa trên cổng huấn luyện (đặt trước FD), không dựa trên FD. Checkpoint huấn luyện 5000 bước (không phải 10000 như mục 2). D2 không được sửa lại cho các lượt sau.

---

## Kết quả 13 (2026-10-08 16:3x, đồng hồ): checkpoint bước 5000 ĐỦ ĐIỀU KIỆN; PINN của C4 = `runs/F13/B3W0s_c/B3W0s_8460306c.pt`

**Quy trình thực tế (công khai):** lượt chạy khởi động 15:42; D2 kích hoạt ở bước 5000 (16:10:31), tiến trình bị dừng (A1); ngoại lệ A2 do người dùng chọn; `runner.eval_stage` chạy trên `last.pt` (bước 5000) cho `run.json`. Tốc độ: 0.23 s/bước ở các bước 1000–4000, đoạn 4000–5000 chậm hơn (8 phút 40 giây). Lượt này **không chạy hết 10000 bước**, nên dự đoán "hết 10000 bước trong 60 phút" không đánh giá được.

**Điều kiện đủ điều kiện (mục 3, không đổi):** cổng đầy đủ **đạt** (a: E_res +2.4 mJ, E_id −4.1 mJ; b: T_min 296.6 K, hụt 0.027%; c: tỉ lệ năng lượng 0.930; check_physics 0 cảnh báo); phần dư PDE 0.008666 ≤ 0.01748. → **Đủ điều kiện, là PINN của C4.** (C3: F08, đạt cổng; quy tắc mục 1.)

**FD (một lần, `experiments/verif_table.py`, FD 10 µm cho trường, 5 µm cho bể chảy, 70% quét):**

| C4 | L2 vùng nóng | RMS nóng | Lệch TB nóng | Độ sâu | Độ rộng | Chiều dài |
|---|---|---|---|---|---|---|
| PINN gốc | 22.5% | 270 K | −195 K | −8.3% | −15.3% | −33% |
| F08 (trượt cổng b) | 17.2% | 206 K | −126 K | −1.0% | −5.4% | −24% |
| **F13 bước 5000 (PINN của hồ sơ)** | **18.9%** | 226 K | −149 K | **−0.7%** | **−5.7%** | **−26%** |

ΔD (nêm − vết thường, độ sâu 70%): PINN của hồ sơ (C4 F13 116.8 − C3 F08 90.4) = **+26.4 µm**; FD 5 µm = 117.60 − 90.83 = **+26.8 µm**. Chỉ báo chu trình nhiệt (ΔT_pk, dwell, tỉ số làm nguội) **chưa tính** cho PINN này.

**Chấm dự đoán (mục 4):**

| Dự đoán | Kết quả |
|---|---|
| đạt cổng (b) sau 10000 bước ~60% | đạt ở bước 5000 (không chạy đến 10000) |
| đạt (ii) PDE ≤ 0.01748 ~75% | **đúng** (0.00867) |
| đủ điều kiện (cả hai + a, c) ~45% | **đúng** (nhờ ngoại lệ A2) |
| độ sâu 70% trong ±3% FD ~75% | **đúng** (−0.7%) |
| L2 vùng nóng 13–20% ~70% | **đúng** (18.9%) |
| lệch TB nóng ≤ −100 K ~85% | **đúng** (−149 K) |
| tốc độ ≤ 0.25 s/bước ~65% | một phần: 0.23 trong 4000 bước đầu, đoạn cuối 0.52 |
| hết 10000 bước trong 60 phút ~80% | không đánh giá được (dừng ở 5000) |

**Kết luận được và không được.** Phạt top-k làm cổng (b) đạt (hụt 1.26% → 0.027%, số điểm dưới T_amb − 5 K từ 7208 xuống 0) mà không phá phần dư PDE. Độ chính xác so với FD tương đương F08, hơi kém ở trường nhiệt (L2 18.9% so với 17.2%; lệch −149 so với −126 K). **Không sửa** được lệch lạnh vùng nóng, độ dài bể chảy (−26%) hay lệch ấm vùng lạnh xa chùm; đúng như dự đoán. **Giới hạn:** một seed, một lần chạy, 5000 bước; ngoại lệ A2 chọn sau khi thấy số cổng (b); số hạng top-k là thay đổi loss; không so được ý "do phạt top-k" với "do huấn luyện thêm 5000 bước" vì không có đối chứng huấn luyện thêm 5000 bước không top-k.
