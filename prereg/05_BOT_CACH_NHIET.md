# Đăng ký trước 05 — bột cách nhiệt có lịch sử nóng chảy (bước 1 sau M3)

Ngày: 2026-10-06 19:54 (đồng hồ). Người dùng đồng ý bước 1 (sau phân tích ở cuối `02` mục 19) và yêu cầu **chạy mốc test trước khi chạy đầy đủ**. **Commit file này trước khi viết mã và trước khi chạy bất kỳ mốc nào.** Mọi thay đổi sau đó qua một mục sửa đổi có giờ lấy từ đồng hồ. Ngưỡng đánh dấu **[ĐỀ XUẤT]** do tôi đề xuất. Làm theo `CLAUDE.md`.

## 1. Mục tiêu và lý do

Mô hình hiện tại (B3, M2″) cho bột **dẫn nhiệt như vật rắn** (`kap_p = 1`); bột chỉ khác ở nhiệt dung (0.55). Do đó ΔD = +9.0 µm là hiệu ứng của "ít khối nhiệt hơn", không phải hiệu ứng cách nhiệt, cơ chế chính của overhang (`02` mục 19, giới hạn 2). Bước này thêm cách nhiệt của bột: **k_bột = r·k_rắn, r = 0.02346 (k_eff = 0.516 W/m·K, tương quan Moser 2019 Eq. 12, đã có trong `sc['pw']['k_ratio']`)**, nhưng **chỉ ở nơi chưa từng nóng chảy**. Lần thử A.1 cũ giữ k thấp cả sau khi bột chảy nên rò nhiệt ngang sai 25%; k theo nhiệt độ hiện tại (`sc['pm']`) sai ở đuôi đã nguội. Cách làm ở đây không thêm biến và không đổi PDE: chỉ đổi hàm hệ số dẫn nhiệt, như `powder_k`.

**Câu hỏi:** với bột cách nhiệt có lịch sử nóng chảy, (i) PINN có qua cổng trung thực (E_res, E_id ≤ 10 mJ; hụt T_min ≤ 1%; tỉ lệ ≤ 1; `check_physics` 0 [WARN]) không; (ii) vòng lặp mặt nạ có hội tụ không; (iii) ΔD, độ xuyên P, T_pk và tốc độ nguội của nêm so với vệt thường đổi bao nhiêu so với mô hình bột dẫn nhiệt như đặc (+9.0 µm, +148 K, +11.5%)?

## 2. Định nghĩa khóa

**Hệ số dẫn nhiệt tĩnh (không có hiệu ứng k_melt):** κ_s(x, y, z, t) = 1 + (r − 1) · s(y, z) · (1 − M(x, y, z, t)), s = `powder_index` (0 rắn → 1 bột, có nêm). Dẫn nhiệt tổng = κ_s · k_melt(θ) (như cũ). Nhiệt dung C* **không đổi** (bột 0.55 cả sau khi chảy): một thay đổi mỗi lần; nhiệt dung của bột đã đông đặc (đúng ra về 1.0) ghi là giới hạn.

**Mặt nạ "đã chảy":** M = σ((x_front(t) − x)/w_x) · e(y, z), x_front(t) = x_laser(t) + 47 µm (mũi bể chảy cố định ở +47 µm trong mọi lần chạy, `02` mục 18), w_x = 8 µm, σ logistic. **Bao bể chảy** e(y, z) = σ((T_max(y, z) − T_sol)/w_T), w_T = 50 K, trong đó T_max(y, z) = max T(x, y, z, t) trên lưới (x − x_laser) ∈ [−300, +60] µm bước 10 µm và t ∈ {0.55, 0.70, 0.85}·t_end của **trường của vòng trước**; tính trên lưới (y, z) 2 µm toàn miền, nội suy song tuyến (`grid_sample`). Ý nghĩa: phía trước mũi bể (x > x_front) vẫn là bột; sau mũi bể, các điểm nằm trong bao bể chảy là kim loại đặc (k_rắn), ngoài bao vẫn là bột (kể cả lớp bột hai bên và vùng bột dưới phần nhô).

**Vòng lặp mặt nạ (ngoài):** vòng 0 dùng bao bể chảy từ trường **không cách nhiệt** hiện có (vệt thường: B3; nêm: M2″); huấn luyện công thức M2″ (Adam, loss = L_PDE + L_top + 1·L_E + 1000·L_P, kiểm cổng đầy đủ mỗi 500 bước, dừng khi hai điểm kiểm liên tiếp đạt và đã qua 2,000 bước, tối đa 6,000 bước, trần 45 phút); khởi động từ trọng số B3 (vệt thường) và M2″ (nêm). Sau vòng j tính bao bể chảy mới từ trường mới; **hội tụ** khi so với bao dùng để huấn luyện: độ sâu bao tại y = 0 lệch ≤ 2 µm, nửa rộng bao trên mặt lệch ≤ 3 µm, và diện tích hiệu đối xứng ≤ 5% diện tích bao [ĐỀ XUẤT]. Chưa hội tụ thì huấn luyện lại (khởi động từ trọng số vòng trước) với bao mới, tối đa **3 lần huấn luyện lại** (vòng 1–3); hết mà chưa hội tụ thì báo "không hội tụ" và dùng vòng cuối với cờ.

**Thay đổi mã (cho mọi bậc có mặt nạ):** hàm `pde`, `top_res` của `legacy/` được thay trong tiến trình bằng bản có mặt nạ (không sửa `legacy/`; dựa vào việc `lpbf_pinn` tra cứu `pde`, `top_res` ở không gian tên mô-đun khi gọi); đạo hàm `∇κ_s` tính một lần trên bản sao tách đồ thị, `div(κ_s K ∇θ) = κ_s Σ_j ∂_j(K g_j) + K g·∇κ_s`; cổng E_id dùng cùng κ_s (thông lượng vách, nhất là vách y = ±Ly/2 qua lớp bột). Bậc mới `B3i` (vệt thường), `B3W0i` (nêm), tag gồm băm bao bể chảy.

## 3. Mốc test S2.17 (chạy trước; trượt mục nào thì dừng, báo cáo, không chạy phần sau)

| ID | Kiểm | Tiêu chí |
|---|---|---|
| (a) | Tương đương: r = 1 (hoặc e ≡ 0 và r = 0.02346 so với `L.pde` với `sc['kap_p'] = r`): phần dư PDE và `top_res` ở float64, 2^14 điểm ngẫu nhiên (cả dải mặt phân cách, nêm, mặt trên) | sai tương đối lớn nhất / RMS ≤ 1e-10 |
| (b) | Đạo hàm: `∇κ_s` bằng autograd so với sai phân trung tâm của `κ_s` (float64, bước 1e-3 µm), trên 2^12 điểm, loại điểm cách biên ô lưới < 3 bước | phân vị 99% của sai số tương đối ≤ 1e-4 |
| (c) | `div(κ_s K ∇θ)` của mã mới (trường giải tích trơn θ(x), K = `k_melt`) so với sai phân của thông lượng tính trực tiếp, float64, 2^12 điểm | phân vị 99% sai số tương đối ≤ 1e-5 |
| (d) | Hình dạng bao bể chảy từ B3: độ sâu bao tại y = 0 và nửa rộng bao trên mặt so với `indicator` (M1) | độ sâu 90.8 ± 3 µm, rộng (hai phía) 167 ± 4 µm |
| (e) | Bản đồ κ_s tại điểm cho trước (mặt nạ từ bao của B3; nêm: M2″): sau mũi bể trong bao, độ sâu 30 µm → ≈ 1; **phía trước mũi** (x_laser + 300 µm), độ sâu 30 µm → ≈ r; ngoài bao (y = 200 µm), sau mũi, độ sâu 30 µm → ≈ r; nền rắn (độ sâu 150 µm) → 1; nêm: dưới phần nhô (bột) → r | mỗi điểm trong ± 5% |
| (f) | E_id của M2″ khi **không** cài mặt nạ = −4.6 ± 0.3 mJ và khi cài mặt nạ với r = 1 = cùng giá trị ± 0.3 mJ (hồi quy cổng) | đúng |
| (g) | Khói `runner.py` cho `B3W0i` (Adam 40 bước, kiểm 20, `--min-steps 20`, thư mục tạm, bao từ M2″) | `run.json` DONE, tag có băm bao, loss hữu hạn; `legacy/ckpt` không đổi |
| (h) | **Canary tối ưu hóa** (nêm và vệt thường, từ trọng số M2″ và B3, 500 bước Adam, công thức M2″, mặt nạ vòng 0): loss khởi điểm [L_PDE, L_top], loss ở bước 500, giây/bước | không NaN/Inf; tổng L_PDE + L_top ở bước 500 ≤ 0.7 × bước 0; ≤ 0.25 s/bước; ghi E_res, E_id, hụt T_min, tỉ lệ (không phải điều kiện) |
| (i) | Hồi quy: tầng 2 (S2.1–S2.4), S2.13 và S2.16(b) chạy lại với mã mới | đạt |

Trần GPU của mốc test: 20 phút (dừng ở 30). Dự đoán: (a)–(g) đạt cả ngay lượt đầu 70%; (h) vệt thường đạt 75%, nêm đạt 55% (nhảy hệ số 43 lần trong vùng nhỏ, loss khởi điểm sẽ cao); (i) đạt 85%.

## 4. Lượt đầy đủ F (chạy nếu S2.17 đạt cả chín; người dùng đã yêu cầu chạy đầy đủ sau mốc test)

Vệt thường `B3i` và nêm `B3W0i` theo vòng lặp mặt nạ ở mục 2, thứ tự: vệt thường trước (nếu vệt thường **trượt cổng** thì dừng và báo cáo: công thức không hợp lệ cho vật lý mới), rồi nêm. Mỗi lần huấn luyện **ĐẠT** theo tiêu chí M2″ (dừng vì hai điểm kiểm liên tiếp đạt; cổng đầy đủ cuối đạt; `check_physics` 0 [WARN]; ≤ 75 phút). Trần tổng 120 phút (dừng ở 180). **Trượt thì dừng, báo cáo, không chạy lại.**

**Đầu ra khi cả hai hình học ĐẠT:** bảng chỉ báo ở 70% (như `02` mục 19) cho vệt thường và nêm cách nhiệt: D, P, T_pk, thời gian lưu, tốc độ nguội, rộng (W₊, W₋); Δ = nêm − thường; so với mô hình bột dẫn nhiệt như đặc. Chiều dài chỉ ghi (quy tắc R1). Nhiễu: chưa có M3 cho mô hình mới; mọi Δ ghi là **một lượt**; M3 cho mô hình mới là bước sau, hỏi người dùng.

**Dự đoán (ghi trước, có thể sai).** (1) Vệt thường cách nhiệt qua cổng 55%, nêm 40%. (2) Mặt nạ hội tụ trong ≤ 3 lần huấn luyện lại, cho cả hai hình học đã qua cổng: 45%. (3) Nếu cả hai qua cổng: ΔD(cách nhiệt) > +9.0 µm 85%; ≥ +20 µm 55%; ≥ +40 µm 20%. (4) Vệt thường cách nhiệt: độ sâu đổi < 5 µm so với B3 (90.8 µm) 60% (bột cách nhiệt chủ yếu ảnh hưởng mép bể và vùng dưới phần nhô). (5) Tổng GPU: mốc test 10–20 phút, lượt đầy đủ 40–100 phút.

## 5. Giới hạn (khóa)

Mặt nạ là xấp xỉ **tĩnh theo bao bể chảy cực đại** (không theo dõi lịch sử nóng chảy từng điểm; vệt một đường, bể gần ổn định); nhiệt dung bột không đổi sau khi chảy; k bột lấy từ tương quan ngoại suy cho bột kim loại sáng (không đo cho IN718 ở dự án này; ±11% của bài gốc chưa tính); lớp bột dưới phần nhô được coi là bột khô ở mọi nhiệt độ (không có thiêu kết); vẫn một điều kiện (300 W, 0.8 m/s), một góc (45°), α_s = 0.35; không có dữ liệu thực nghiệm; kết quả là **trong mô hình này**. Cờ `sc['pm']` (k theo nhiệt độ) không dùng.
