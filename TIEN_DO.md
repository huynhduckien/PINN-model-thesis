# Tổng kết tiến độ — PINN overhang không support (LPBF)

Cập nhật: 2026-10-07 16:00 (đồng hồ). Mục đích: một nơi duy nhất để kiểm tra những gì **đã đạt** và **chưa đạt**, và để khởi động lại công việc ở một cuộc hội thoại mới. Mọi số lấy từ file kết quả gốc (`runs/**/run.json`, `report.json`, `summary.json`) và các đăng ký trước trong `prereg/`; chỗ nào chưa tính lại độc lập thì ghi rõ ở mục 6. Cách gọi đúng của mô hình: xem mục 2.

## 1. Mục tiêu và ràng buộc đã chốt

- PINN thuần vật lý (DeepXDE, backend pytorch): **không FD, không dữ liệu trong loss**.
- Một điều kiện: 300 W, 0.8 m/s, bán kính chùm 40 µm, lớp bột 60 µm, IN718, **một đường quét đơn** dọc mép nêm 45° (đỉnh nêm y = 0).
- Đầu ra: chỉ báo nhiệt **tương đối** nêm − vệt thường (độ xuyên dưới đáy lớp bột, tích tụ nhiệt, tốc độ nguội); không dự đoán dross hay võng.
- Quy trình: đăng ký trước (tiêu chí khóa trước khi chạy), mốc test trước lượt đầy đủ, sửa đổi có giờ lấy từ đồng hồ, không nới ngưỡng sau khi thấy kết quả.
- Người dùng quyết **không làm S0.8** (bảng số thực nghiệm); do đó không có hiệu chỉnh M5 và không có neo ngoài.

## 2. Cách gọi đúng (bắt buộc trong báo cáo, bản thảo, luận văn)

*Mô hình PINN nhiệt (dẫn nhiệt + ẩn nhiệt nóng chảy + bay hơi) cho một đường quét đơn dọc mép phần rắn có mặt nghiêng 45° hướng xuống, trong miền hữu hạn 1500 × 480 × 240 µm, bột cách nhiệt với mặt nạ vùng đã chảy, độ rộng chuyển tiếp hệ số dẫn nhiệt kiểm ở 30 và 45 µm, một điều kiện công nghệ, chưa có dữ liệu ngoài.* **Không** gọi là "mô phỏng quá trình in overhang" hay "kết quả support free". Chi tiết phạm vi: `prereg/05` mục 18 và `prereg/06` mục 14.

## 3. Đã đạt

| Mục tiêu | Kết quả | Nguồn |
|---|---|---|
| Hạ tầng và mốc test tầng 0–3 (S2.1–S2.20) | đạt (S2.20 đạt lượt lại sau một lỗi khung kiểm và một lần máy chạy chậm) | `prereg/01`, `05`, `06`; `smoke/` |
| Cổng năng lượng trung thực: \|E_res\| ≤ 10 mJ **và** \|E_id\| ≤ 10 mJ; hụt T_min ≤ 1%; (tích + bay hơi)/nạp ≤ 1 | đạt; **phát hiện cắt θ ≥ 0 che 16–19 mJ nguồn âm** nên bỏ M2′ | `prereg/03` |
| M1: vệt thường làm mốc (B3) | đạt 9/9 cổng; sâu 90.78 µm, rộng ≈ 167 µm | `runs/M1/indicator.json` |
| M2: nêm qua cổng, bột dẫn nhiệt như đặc (M2″) | đạt; E_res −5.4, E_id −4.6 mJ; **ΔD = +8.99 ≈ +9.0 µm** (cận dưới) | `runs/M2pp/run.json` |
| Bột cách nhiệt có lịch sử nóng chảy, mép mềm 30 µm | đạt cả vệt thường (`B3s`) và nêm (`B3W0s`); mặt nạ hội tụ | `runs/Fs/`, `prereg/05` mục 13 |
| M3 mô hình cách nhiệt | **ỔN ĐỊNH** (3 cặp hợp lệ); nhiễu ΔD ±0.41 µm, ΔT_pk ±10.9 K, Δlưu ±10 µs, tỉ số nguội ±0.04; ΔW không tách khỏi nhiễu | `runs/M3s/report.json`, `prereg/05` mục 17 |
| M4: bảng chỉ báo nêm − vệt thường kèm nhiễu và giới hạn | đạt (bảng ở mục 4) | `prereg/06` mục 14 |
| Độ nhạy theo độ rộng chuyển tiếp | đạt một phần: **30 và 45 µm**; hướng hiệu ứng không đổi; ΔD > +9.0 µm ở cả hai | `runs/Fw/summary.json`, `prereg/06` mục 13 |
| Hình chuẩn (mặt cắt ngang tại x = x_laser − 73 µm, 70% quét; vệt thường \| nêm; mỗi ô thang màu riêng) và sơ đồ hình học | đạt | `runs/Figures/`, `plot_melt_pool.py`, `plot_geometry.py` |
| Phạm vi mô hình và cách gọi đúng | đạt | `prereg/05` mục 18, `prereg/06` mục 14 |

## 4. Bảng chỉ báo cuối cùng (nêm − vệt thường; 70% quét cho hình học, 50% quét cho chu trình nhiệt)

| Chỉ báo | Giá trị | Ghi chú |
|---|---|---|
| độ sâu = độ xuyên ΔD = ΔP | **+16.5 … +18.3 µm** (độ rộng chuyển tiếp 45 … 30 µm) | nhạy vừa; nhiễu huấn luyện ±0.4 µm (đo ở 30 µm); bột như đặc +9.0 µm |
| ΔT_pk tại đáy lớp bột | **+160 ± 11 K** (30 µm); +142 K (45 µm) | trong nhiễu, sát biên |
| Δ thời gian lưu trên T_sol | **+222 ± 10 µs** (30 µm); +221 µs (45 µm) | không nhạy |
| tỉ số tốc độ nguội nêm/thường | **0.54 … 0.68** | nhạy; luôn < 1 |
| **Không dùng làm chỉ báo** | chiều dài (R1), ΔW (−0.9 / −1.4 µm, nhiễu 1.9 µm), nhiệt độ đỉnh tuyệt đối (~5,500 K > T_vap 3,120 K, không có áp suất giật/keyhole) | |

Độ sâu từng hình học (µm): vệt thường 89.5 (30 µm) / 90.2 (45 µm); nêm 107.8 (30 µm) / 106.7 (45 µm).

## 5. Chưa đạt

| Mục | Vì sao | Ghi chú |
|---|---|---|
| M3 theo tiêu chí gốc (không cách nhiệt) | KHÔNG ỔN ĐỊNH, chỉ do tiêu chí lát thời gian của chiều dài | phán quyết giữ nguyên (R1); chiều dài bị loại khỏi chỉ báo |
| M5 hiệu chỉnh, mọi neo thực nghiệm | không có dữ liệu ngoài; người dùng quyết không làm S0.8 | mọi kết quả là "trong mô hình này"; chưa biết sai số so với thực tế |
| Tiêu chí in được (dross, võng, độ xuyên tới hạn) | chưa có | nên chưa kết luận được "support-free" |
| Độ rộng chuyển tiếp W < 30 µm | 15 µm không huấn luyện được (E_id −21.1 mJ sau 6,000 bước); 8 µm cũng không | giới hạn của công thức tối ưu (Adam, mạng 6 × 24), không phải kết luận vật lý; vùng hẹp chưa kiểm |
| Độ nhạy theo hấp thụ α_s (hướng 4) | dừng sau 2 trong 7 lượt (chỉ có vệt thường); ΔD(α_s) chưa đo | `prereg/04` mục 9; chạy lại được bằng `experiments/h4_alpha.py run` |
| Độ sâu miền và đáy giữ nhiệt độ môi trường | chưa kiểm | đáy cách đáy bể ~140 µm ≈ 1.7 độ dài khuếch tán; ước lượng thô, chưa chứng minh |
| Sức chứa mạng (R3), L-BFGS (S1) | chưa kiểm | khớp PDE cách nhiệt còn thô (L_PDE ~1e-2, gấp ~10 lần B3); biên E_id mỏng ở 30 µm (−8.8 mJ vệt thường) |
| Mép mũi bể w_x = 8 µm | không quét | chỉ quét hệ số chung cho mặt phân cách lớp bột và mép bao |
| Góc nêm ≠ 45°, vị trí chùm ≠ đỉnh nêm, công suất/tốc độ khác | ngoài phạm vi "một điều kiện" | cần nới ràng buộc để nói về thiết kế |
| Kết khối (M6) | không nằm trong kế hoạch | mô hình không có biến dạng hình học |
| Hai hình cũ `runs/M1/B3_melt_pool_*.png` | quy chuẩn cũ; chưa xóa (chờ quyết định) | không nằm trong git; xóa là không khôi phục được |

## 6. Giới hạn của chính phép kiểm này (đọc trước khi dùng số)

- Đã tính lại từ file gốc: cổng và E của các lần huấn luyện cho số chính; ΔD bột như đặc (8.99 µm); báo cáo M3 cách nhiệt; bảng độ nhạy (`runs/Fw/summary.json`).
- **Chưa tính lại độc lập**, lấy từ văn bản prereg: các số về pha bể chảy (đông đặc sau khoảng 575–610 µs, sâu nhất 90–200 µs sau chùm), kết quả tầng 0 (S0.x), bản đồ vật liệu nêm S0.5.
- M3 cách nhiệt "ổn định" theo bộ tiêu chí mới đã khóa trước khi thấy số liệu, **khác** bộ tiêu chí gốc (bỏ tiêu chí lát thời gian của chiều dài, vì R1 đã loại chiều dài); ba ngưỡng 15 K, 50 µs, 0.10 là đề xuất của tôi. Nhiễu huấn luyện đo ở 30 µm; ở 45 µm (một lượt) chưa đo, dùng nhiễu của 30 µm là giả định.
- Hai điểm độ rộng (30 và 45 µm) không xác định dạng hàm; không nội suy; không có kết luận cho W < 30 µm. Xu hướng ΔD giảm khi W tăng khớp dự đoán cơ chế nhưng chưa chứng minh.
- ΔT_pk ở 45 µm lệch 17.9 K: dưới ngưỡng "trong nhiễu" (21.8 K) nhưng trên dung sai 15 K của M3; phân loại sát biên.
- **Nhiễu là nhiễu quy trình huấn luyện, không phải sai số so với thực tế.**
- Lỗi của chính tôi đã ghi trong prereg và đã sửa: khẳng định sai "float64 chậm 29 lần" (thật ra là bộ nhớ GPU, `01f`); baseline E_res sai (`01g`); bản cắt θ ≥ 0 che lỗi nguồn âm (`03`); bốn lỗi đặc tả điểm kiểm (không nới ngưỡng); khẳng định sai "tiêu chí lượt đầy đủ không phụ thuộc tốc độ" (có trần 45 phút theo đồng hồ); lời giải thích VRAM cho lần chạy chậm chỉ là nghi phạm (nguyên nhân thật chưa rõ, máy chậm 4 lần 13:03–14:25 rồi tự hồi phục).

## 7. Hướng đi tiếp theo (người dùng quyết)

1. **Viết kết quả** với đúng phạm vi ở mục 2 và bảng mục 4, không thêm thí nghiệm. Chi phí thấp; chỉ nói được "so sánh tương đối trong một mô hình".
2. **Quét góc nêm** (ví dụ 30° và 60° cùng quy trình; nới "một điều kiện"). Cho xu hướng theo góc để đối chiếu **định tính** với kinh nghiệm đã công bố (góc nhỏ thì dross và tích nhiệt nặng hơn), không cần dữ liệu thực nghiệm trong loss; gần mục tiêu thiết kế nhất. Cần thêm tham số góc cho hình học; ước thô 3–5 giờ GPU sau một mốc test (chưa đo). **Cần đăng ký trước.**
3. **Độ vững kỹ thuật:** độ sâu miền; L-BFGS cho cấu hình mềm (có thể mở được 15 µm); sức chứa mạng. Chi phí cao; chưa thấy dấu hiệu đổi kết luận.
4. **Đối chiếu độ sâu/độ rộng vệt thường với số liệu đã công bố** (không dùng dữ liệu trong loss, không chạy S0.8). Chi phí rất thấp, chỉ là điểm tham khảo thô.

**Đề xuất của tôi:** hướng 1 song song với hướng 2; hoãn hướng 3 trừ khi người phản biện đòi. Câu hỏi để quyết: luận văn cần "xu hướng theo góc" để nói về thiết kế, hay dừng ở "hiệu ứng nêm tại 45° và độ nhạy của nó"?

## 8. Bản đồ file để khởi động lại

- **Quy trình và tiêu chí khóa:** `prereg/00_KE_HOACH_CHAY_OVERHANG.md` (kế hoạch, M0–M6), `01` (tầng 0–3), `02` (M2–M4 nêm bột như đặc; mục 19, 20 bảng cuối và đính chính chu trình nhiệt), `03` (M2′ cắt θ ≥ 0 trượt; M2″), `04` (hướng 4, dừng giữa chừng), `05` (bột cách nhiệt; mục 17 M3, mục 18 phạm vi, mục 19 trỏ tới 06), `06` (độ nhạy độ rộng chuyển tiếp; mục 13 kết quả, mục 14 bảng và phạm vi cập nhật).
- **Mã:** `config.py` (RUNS, scales, build), `runner.py` (run/supervise/watch), `gates.py`, `energy.py`, `maskphys.py`, `analysis.py`, `indicator.py`, `lock.py` (khóa GPU), `legacy/lpbf_pinn.py` (đóng băng, không sửa; sha 16 `b0f5955a2385e79c`).
- **Thí nghiệm:** `experiments/f_insul.py` (vòng lặp mặt nạ; `--soft`, `--width W`), `m3_ins.py`, `g_width.py`, `h4_alpha.py`; kiểm: `smoke/tier0…tier2l.py`.
- **Kết quả chính:** `runs/M1` (B3), `runs/M2pp` (M2″), `runs/Fs` (cách nhiệt 30 µm), `runs/Fw45`, `runs/Fw15`, `runs/Fw/summary.json`, `runs/M3s/report.json`, `runs/ENV` (bao vòng 0).
- **Hình:** `runs/Figures/` (sơ đồ hình học, mặt cắt ngang) bằng `plot_geometry.py`, `plot_melt_pool.py`.
- **Lệnh mẫu:** hình chuẩn `python plot_melt_pool.py runs/Figures insulated B3s runs/Fs/B3s_it0/B3s_d2f9f1db.pt B3W0s runs/Fs/B3W0s_it1/B3W0s_ba6419f6.pt --x-offset -73`; tổng hợp độ nhạy `python experiments/g_width.py summarize`.
- **Luật làm việc:** không dùng `git add -A` (thêm file theo tên); chỉ một phiên ghi/commit; đăng ký trước, không nới tiêu chí; kiểm tốc độ (~0.17 s/bước) và `nvidia-smi` trước khi chạy dài (trần thời gian của runner theo đồng hồ có thể biến "đạt" thành "không đạt" giả); GPU một tiến trình một lúc (`lock.py`); thư mục `tailieu/` là của người dùng, chưa vào git.
