# Đăng ký trước 10 — thăm dò sức chứa mạng: mạng 6×24 có biểu diễn được nghiệm FD của PINN cách nhiệt không?

Ngày: 2026-10-08 11:56 (đồng hồ). Người dùng đồng ý thử hướng này sau `prereg/09`. **Commit file này trước khi viết mã và trước khi chạy bất kỳ lượt nào.** Mọi thay đổi sau đó qua một mục sửa đổi có giờ lấy từ đồng hồ. Ngưỡng đánh dấu **[QUY ƯỚC]** là lựa chọn của tôi, người dùng quyết. Làm theo `CLAUDE.md`.

## 1. Câu hỏi và phạm vi

`prereg/08` và `09` cho thấy sai số vùng sau chùm của PINN cách nhiệt (C3 −152 K, C4 −193…−232 K) không đáp ứng với số bước huấn luyện (loss giảm theo bước^(−0.25)) lẫn mật độ điểm lấy mẫu. Nghiệm FD thỏa PDE đến sai số rời rạc nên là một nghiệm có loss gần 0; vì vậy PINN bị giới hạn bởi **tối ưu** (không tìm được nghiệm) hoặc **sức chứa** (mạng không biểu diễn được nghiệm). Hai trường hợp dẫn tới hai hướng khác nhau: (a) sức chứa thiếu → cần mạng lớn hơn; (b) sức chứa đủ → cần cải thiện tối ưu (L-BFGS, điều kiện hóa).

**Câu hỏi:** mạng của PINN hiện tại (FNN 6 lớp ẩn × 24 nơ-ron, kích hoạt `sin`, cùng biến đổi đặc trưng và mặt nạ cứng đầu ra) khi được **khớp có giám sát** vào trường nhiệt FD (4D: x, y, z, t) của C3 và C4, đạt sai số bao nhiêu? Nếu thấp thì sức chứa đủ; nếu cao như sai số của PINN thì sức chứa là giới hạn. Phụ: một mạng rộng hơn (6 × 64) có khớp tốt hơn không?

**Đây là phép chẩn đoán, không phải huấn luyện PINN.** Khớp có giám sát dùng dữ liệu FD trong loss, nên **không được dùng cho PINN của hồ sơ** và không thay đổi quy tắc "không FD, không dữ liệu trong loss" của PINN (`TIEN_DO.md` mục 1). Mạng khớp chỉ để đo năng lực biểu diễn. Kết luận "sức chứa đủ" **không** chứng minh PINN tìm được nghiệm đó bằng PDE (khớp có giám sát dễ tối ưu hơn nhiều), chỉ nói nghiệm biểu diễn được. Kết luận "sức chứa giới hạn" mạnh hơn: nếu ngay cả khớp trực tiếp cũng không đạt thì huấn luyện bằng PDE càng không.

## 2. Thiết kế (khóa)

**Dữ liệu:** FD `fdv2` lưới 10 µm (cùng cấu hình `fd_cases`) của C3 (`B3s`, bao T_max `runs/ENV/B3_iter0.pt`) và C4 (`B3W0s`, bao T_max `runs/Fs/B3W0s_it0/envelope_next.pt`), ảnh chụp T tại thời điểm cách đều **10 µs, lệch pha 5 µs** (t_k = 5 + 10k µs, k = 0…149). Ba thời điểm đánh giá 750, 1050, 1350 µs (50%, 70%, 90% quét) nằm đúng giữa hai ảnh chụp huấn luyện nên **không có trong tập huấn luyện** (thử nội suy theo thời gian). Mục tiêu khớp: θ = (T − T_amb)/T_c (như đầu ra mạng).

**Mạng và khớp:** hai nhánh, cùng cách dựng `legacy.build_net` (có biến đổi đặc trưng và mặt nạ cứng):
- **A (hiện tại):** `depth = 6`, `width = 24`, `act = sin`, `init = Glorot normal`, khởi tạo ngẫu nhiên (seed 42).
- **B (rộng hơn):** như A nhưng `width = 64`.
Hàm mất mát: sai số bình phương trung bình của θ trên mẫu lấy ngẫu nhiên mỗi bước, 50% từ các ô nóng (T_FD > 600 K) và 50% đều từ mọi ô (thời gian lấy ngẫu nhiên trong tập huấn luyện); Adam float32, `lr = 2×10⁻³` giảm cosin xuống `10⁻⁵`, **[QUY ƯỚC] 30000 bước**, kích thước lô **[QUY ƯỚC] 2¹⁹**. Một seed. Mỗi (ca, nhánh) một lần khớp, không chọn lại theo kết quả. Ghi đường cong loss; nếu loss còn giảm nhanh ở cuối (độ dốc 5000 bước cuối trên 2% mỗi 1000 bước), kết quả ghi là **cận dưới của năng lực khớp** (khớp chưa hội tụ).

## 3. Đo và phân loại (khóa)

**Đo (mạng khớp so với FD 10 µm, tại ba thời điểm giữ lại, chỉ các ô T_FD > 600 K):** (i) RMS_nóng = căn trung bình bình phương của (T_mạng − T_FD) [K]; (ii) trung bình (T_mạng − T_FD) theo vùng của `fd_errormap.py` (vùng quanh chùm, sau chùm); (iii) từ trường mạng khớp, tính cùng cách `analysis.net_indicator`: độ sâu ở 70%, chu trình nhiệt ở 50% (T_pk, thời gian lưu), rồi ΔD, ΔT_pk, Δ thời gian lưu = (C4 − C3), so với FD (+26.77 µm, +179.9 K, +230.3 µs). **Mốc so sánh với PINN F08** (`errormap_f08.json`, tính RMS_nóng tổng từ RMS và số ô từng vùng ở 70%): sẽ tính trước khi chạy khớp và ghi vào mục kết quả.

**Phân loại cho nhánh A (mạng hiện tại), theo cả hai ca:**
- **SỨC CHỨA ĐỦ:** RMS_nóng ≤ 25% RMS_nóng của PINN F08 và \|trung bình vùng sau chùm\| ≤ 25 K, ở **cả hai** ca **[QUY ƯỚC: 25%, 25 K]**.
- **SỨC CHỨA GIỚI HẠN:** RMS_nóng ≥ 75% RMS_nóng của PINN F08 ở **ít nhất một** ca **[QUY ƯỚC: 75%]** (và khớp đã hội tụ; nếu chưa hội tụ thì ghi KHÔNG KẾT LUẬN).
- **KHÔNG KẾT LUẬN:** còn lại.

**Nhánh B** chỉ báo cáo: nếu A thuộc SỨC CHỨA GIỚI HẠN, B cho biết mạng rộng hơn có cải thiện không (RMS_nóng giảm xuống dưới 50% của A thì coi là "mở rộng có tác dụng").

**Số lần thử:** một lần cho mỗi (ca, nhánh). Hành động tiếp theo (không thuộc 10): SỨC CHỨA ĐỦ → đăng ký 11 cho cải thiện tối ưu (L-BFGS với mẫu cố định, điều kiện hóa); SỨC CHỨA GIỚI HẠN → đăng ký 11 cho mạng lớn hơn (cần huấn luyện lại chuỗi bậc từ đầu hoặc phương án khởi tạo khác).

## 4. Dự đoán (ghi trước, có thể sai)

- Nhánh A: SỨC CHỨA ĐỦ ~25%; SỨC CHỨA GIỚI HẠN ~30%; KHÔNG KẾT LUẬN ~45%. Lý do: mạng 6×24 (~3000 tham số) cho trường 4D có mép chuyển tiếp 3.75–30 µm, bể chảy di động và vùng đuôi dài ~1 mm; không có bằng chứng trước.
- RMS_nóng của nhánh A nằm giữa 30 và 100 K (~60%).
- Nhánh B: RMS_nóng thấp hơn A ít nhất 2 lần (~65%).
- ΔD tính từ mạng khớp A nằm trong 1.83 µm của FD (~50%); ΔT_pk trong 16 K (~25%); Δ thời gian lưu trong 22.2 µs (~25%).
- Khớp chưa hội tụ (độ dốc cuối > 2%/1000 bước) ở nhánh A: ~30%; ở nhánh B: ~40%.
- Tổng thời gian chạy (4 lượt khớp + hai lượt FD) trong 1.5 giờ: ~60%.

## 5. Giới hạn (khóa)

Một seed, một lần khớp mỗi (ca, nhánh); khớp có giám sát không tương đương huấn luyện bằng PDE; FD 10 µm làm mục tiêu (sai số FD 10 µm cỡ vài K ở đuôi và ~0.4 µm ở độ sâu, nhỏ so với 100+ K của PINN); mạng khớp có hàm mặt nạ cứng đầu ra nên khớp ở mép miền bị ràng buộc cố định ở T_amb; kết luận về sức chứa chỉ cho kiến trúc FNN sin với biến đổi đặc trưng hiện tại, không nói về các kiến trúc khác (Fourier features, chia miền). Kết quả này không đổi bảng chỉ báo và không phải PINN của hồ sơ.
