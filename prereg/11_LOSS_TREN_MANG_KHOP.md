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


---

## Kết quả 11 (2026-10-08 13:40, đồng hồ): phân loại theo quy tắc là DẠNG LOSS (L), nhưng phép đo có nhược điểm thiết kế nên không đủ để kết luận

**Khớp lại nhánh A (có lưu trọng số):** tái lập đúng kết quả của 10 (xác định ở mức in ra): C3 RMS nóng 11.3 K (70%: 10.9 K), lệch sau chùm −1.1 K, 643 s; C4 12.7 K (12.6 K), −2.2 K, 313 s. Điều kiện chấp nhận (≤ 25% RMS PINN) thỏa.

**Loss huấn luyện** (cùng mã `runner`/DeepXDE, điểm collocation mặc định, `λ_E = 1`, `λ_P = 1000`; thành phần [phần dư PDE, biên mặt trên, năng lượng, dương]; hai thành phần cuối trung bình 20 lần):

| Ca | PINN F08 | Mạng khớp A | R = PDE(khớp)/PDE(PINN) |
|---|---|---|---|
| C3 | [0.00545, 0.00044, 0.00020, 0.00005] | [4.222, 0.494, 0.0198, 0.0021] | **774** |
| C4 | [0.00874, 0.00036, 0.00025, 0.00004] | [4.051, 0.533, 0.0157, 0.0030] | **464** |

Loss của PINN F08 trùng số đã ghi trong `run.json` (0.00545; 0.00874). **Phần dư tương đối (trung vị / phân vị 90), ~350 điểm gần chùm:** PINN C3 0.141 / 0.431, mạng khớp 0.360 / 2.032; PINN C4 0.184 / 0.685, mạng khớp 0.425 / 1.357.

**Phân loại theo mục 3: DẠNG LOSS (L)** (R ≥ 1.0 ở cả hai ca).

**Nhược điểm thiết kế của chính tôi (phải ghi cùng kết quả):** mạng khớp được tối ưu để khớp **giá trị** nhiệt độ, không khớp đạo hàm, nên đạo hàm bậc hai của nó (vào phần dư PDE) nhiễu và loss rất lớn dù giá trị chỉ lệch 11–13 K. Ước lượng thô trong mục 4 của 11 ("phần dư ~1% nên loss khớp có thể thấp") **sai nhiều bậc độ lớn**. Chênh lệch phần dư tương đối trung vị giữa mạng khớp và PINN chỉ cỡ 2.5 lần, còn loss chênh 500–800 lần: loss bị chi phối bởi đuôi (phân vị 90 của mạng khớp 136–203%), tức vài điểm có phần dư rất lớn. Vì vậy "R ≥ 1.0" **không chứng minh** loss ưu tiên nghiệm sai: nó chỉ cho thấy một mạng chính xác về giá trị nhưng không chính xác về đạo hàm có loss PDE cao, điều hiển nhiên. Phép đo này **không phân biệt được** tối ưu (T) với dạng loss (L). Điều nó cho thấy chắc chắn: loss PDE trên điểm collocation **không phải thước đo của sai số giá trị**; PINN F08 (loss 0.005–0.009, sai số nhiệt độ 135–206 K RMS) có loss thấp hơn hơn 500 lần so với mạng chính xác về giá trị (11–13 K RMS), nên loss thấp không bảo đảm nghiệm đúng.

**Chấm dự đoán (mục 4):**

| Dự đoán | Kết quả |
|---|---|
| TỐI ƯU ~35%, DẠNG LOSS ~35%, KHÔNG KẾT LUẬN ~30% | **DẠNG LOSS** theo quy tắc (nhưng xem nhược điểm thiết kế ở trên) |
| R của C3, C4 cùng phía ngưỡng 1.0 (~65%) | **đúng** (774 và 464) |
| phần dư tương đối trung vị của mạng khớp thấp hơn PINN ≥ 2 lần ở cả hai ca (~45%) | **sai**: cao hơn 2.6 và 2.3 lần |
| RMS nóng khớp lại trong ±30% so với lần khớp ở 10 (~75%) | **đúng** (trùng) |
| tổng thời gian dưới 45 phút (~70%) | **đúng** (~25 phút) |
| (mục 4, lý giải) phần dư của mạng khớp ~1% nên loss thấp | **sai nhiều bậc độ lớn** |

**Hướng tiếp theo (cần đăng ký 12):** cách phân biệt (T) và (L) không bị nhiễu đạo hàm: khởi động từ trọng số mạng khớp rồi **tinh chỉnh bằng chính loss PINN** (không dữ liệu) vài nghìn bước. Nếu loss PDE giảm về mức của PINN (≤ 0.009) **trong khi sai số giá trị so với FD vẫn thấp** (RMS nóng ≤ 30 K, lệch vùng sau chùm ≤ 25 K) thì tồn tại nghiệm chính xác có loss thấp và PINN F08 bị kẹt ở cực trị tồi (T). Nếu loss giảm nhưng trường dịch về nghiệm sai của PINN (lệch vùng sau chùm tiến về −150 K) thì hàm loss ưa nghiệm sai (L). Lưu ý: nếu (T) đúng, khởi động từ mạng khớp dùng dữ liệu FD nên **không** dùng được cho PINN của hồ sơ; kết quả chỉ chỉ hướng (đổi tối ưu, không đổi hàm loss).

**Giới hạn:** một seed, một lần khớp mỗi ca; kết quả nói về PINN F08 (30 µm, mặt nạ cố định, điểm collocation mặc định), không nói về mọi cấu hình; mạng khớp không phải PINN của hồ sơ.
