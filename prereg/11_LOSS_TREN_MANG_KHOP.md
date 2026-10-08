# Đăng ký trước 11 — loss PDE của PINN tại trọng số của mạng khớp: tối ưu hay dạng loss?

Ngày: 2026-10-08 13:10 (đồng hồ). Người dùng đồng ý bước tiếp theo sau `prereg/10`. **Commit file này trước khi sửa mã và trước khi chạy bất kỳ lượt nào.** Mọi thay đổi sau đó qua một mục sửa đổi có giờ lấy từ đồng hồ. Ngưỡng đánh dấu **[QUY ƯỚC]** là lựa chọn của tôi, người dùng quyết. Làm theo `CLAUDE.md`.

## 1. Câu hỏi và phạm vi

`prereg/10` cho thấy mạng của PINN (6×24, 3145 tham số) **biểu diễn được** trường FD của C3, C4 (RMS 11–13 K, so với 135–206 K của PINN F08). Vậy PINN F08 không tìm được vùng nghiệm đó khi huấn luyện bằng phần dư PDE. Có hai cách giải thích:
- **(T) Tối ưu:** mạng khớp có loss PDE (trên cùng các điểm collocation) **thấp hơn rõ rệt** PINN đã huấn luyện; PINN bị kẹt ở cực trị tồi. Cách chữa: tối ưu tốt hơn (L-BFGS với mẫu cố định, tốc độ học, điều kiện hóa).
- **(L) Dạng loss:** mạng khớp có loss PDE **không thấp hơn** (hoặc cao hơn) PINN. Hàm loss "phần dư PDE trên điểm collocation" không phân biệt được nghiệm gần đúng với nghiệm sai. Cách chữa: đổi dạng loss (trọng số theo vùng, chuẩn hóa phần dư, dạng yếu).

**Câu hỏi:** loss huấn luyện của PINN (cùng mã `runner`, cùng điểm collocation mặc định, cùng `λ_E`, `λ_P`) ở trọng số của mạng khớp có thấp hơn ở trọng số của PINN F08 không?

**Đây là phép chẩn đoán.** Mạng khớp dùng dữ liệu FD, không phải PINN của hồ sơ, không dùng để thay PINN. Nó không thay đổi quy tắc "không FD, không dữ liệu trong loss" cho PINN.

## 2. Thiết kế (khóa)

- **Khớp lại nhánh A (6×24) cho C3, C4 và lưu trọng số** (`prereg/10` không lưu). Đúng cấu hình của `prereg/10` (FD 10 µm, 150 ảnh chụp huấn luyện, 30000 bước Adam, lô 2¹⁹, seed 42). GPU không xác định hoàn toàn nên khớp lại **không** trùng bit với 10: báo cáo RMS nóng của khớp lại và ghi chênh lệch so với 10 (chấp nhận nếu RMS nóng ở 70% vẫn ≤ 25% của PINN F08, điều kiện của 10).
- **Đo loss** bằng cùng mã huấn luyện (`runner.build`, float32, `energy_lam = 1`, `pos_lam = 1000`, điểm collocation mặc định `config.SAMPLING`, bao T_max đúng như F08: C3 `runs/ENV/B3_iter0.pt`, C4 `runs/Fs/B3W0s_it0/envelope_next.pt`), bốn thành phần của DeepXDE (phần dư PDE, biên mặt trên, năng lượng, dương) ở:
  - (a) trọng số PINN F08 (`runs/F08/B3s_c/B3s_92803317.pt`, `runs/F08/B3W0s_c/B3W0s_98431da4.pt`), kiểm: phải trùng loss cuối đã ghi (0.00545 và 0.00874) trong sai số lấy mẫu của hai số hạng ngẫu nhiên;
  - (b) trọng số mạng khớp A khớp lại.
  Hai số hạng đầu (PDE, biên) xác định (cùng điểm); hai số hạng sau ngẫu nhiên nên đo 20 lần và lấy trung bình.
- **Phần dư tương đối** (cách `fd_fieldcheck.residual_stats`, ~350 điểm gần chùm, chuẩn hóa theo \|C(1+latent)θ_t\| + \|q\|) cho mạng khớp, cạnh số của PINN F08 (trung vị C3 14.1%, C4 18.4%; phân vị 90 là 43.1%, 68.5%).
- Không huấn luyện thêm PINN trong 11.

## 3. Phân loại (khóa)

Gọi R = loss_PDE(mạng khớp) / loss_PDE(PINN F08), trên cùng điểm collocation, cho từng ca.
- **TỐI ƯU (T):** R ≤ 0.5 ở **cả hai** ca **[QUY ƯỚC: 0.5]**.
- **DẠNG LOSS (L):** R ≥ 1.0 ở **cả hai** ca **[QUY ƯỚC: 1.0]**.
- **KHÔNG KẾT LUẬN / HỖN HỢP:** còn lại.
Báo cáo kèm: loss biên, năng lượng, dương, tổng có trọng số; phần dư tương đối; RMS nóng của mạng khớp.

## 4. Dự đoán (ghi trước, có thể sai)

- Phân loại: TỐI ƯU ~35%, DẠNG LOSS ~35%, KHÔNG KẾT LUẬN ~30%. Lý do không nghiêng: phần dư PDE liên quan đạo hàm bậc hai của sai số khớp (11–13 K, bước sóng ~100 µm) cỡ ~1% của số hạng thời gian theo ước lượng thô, gợi ý loss khớp có thể thấp; nhưng PINN đã trải qua tối ưu hàng chục nghìn bước nên cũng có thể đã ở mức thấp.
- R của C3 và C4 cùng phía ngưỡng 1.0: ~65%.
- Phần dư tương đối trung vị của mạng khớp thấp hơn PINN ít nhất 2 lần ở cả hai ca: ~45%.
- RMS nóng của khớp lại ở 70% trong ±30% so với lần khớp ở 10: ~75%.
- Tổng thời gian dưới 45 phút: ~70% (hai lần khớp nhánh A ~25 phút + FD + đo loss).

## 5. Giới hạn (khóa)

Một seed, một lần khớp mỗi ca; mạng khớp là một điểm trong lân cận nghiệm FD chứ không phải nghiệm đúng; loss của DeepXDE phụ thuộc chuẩn hóa và bộ điểm collocation mặc định (không phải bộ điểm của nhánh W ở 09); kết quả nói về PINN F08 (30 µm, mặt nạ cố định), không nói về mọi cấu hình. Không đổi bảng chỉ báo, không phải PINN của hồ sơ.
