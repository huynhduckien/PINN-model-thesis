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

## 6. Sửa đổi 05a (19:57, sau lượt chạy đầu của S2.17, trước khi chạy lại): lỗi đặc tả điểm kiểm (e) cho nêm

Lượt đầu (19:57): (a), (b), (c), (d) và (e) cho vệt thường **đạt** (a: sai tương đối 0 và 9.3e-15; b: 2.2e-7; c: 5.2e-7; d: bao từ B3 sâu 90.0 µm, rộng 166.0 µm; e vệt thường: 1.000 / 0.023 / 0.023 / 1.000). **(e) cho nêm trượt ở một điểm**: "nền rắn, độ sâu 150 µm tại y = 0" cho κ_s = 0.023 thay vì 1.000. Nguyên nhân là **lỗi của chính điểm kiểm**, không phải của mã: ở nêm y_a = 0, nền rắn nằm ở y ≥ y_a + z_pow − z; tại z = 90 µm (độ sâu 150 µm so với mặt trên 240 µm) điều kiện là y ≥ 90 µm, nên điểm y = 0 là **bột dưới phần nhô**, và κ_s = r là đúng (nó nằm ngoài bao, sâu hơn bao 98 µm). Các điểm nêm khác đạt (dưới phần nhô y = −100 µm: 0.023; nêm rắn y = +100 µm, z = 150 µm: 1.000).

**Sửa (chỉ điểm kiểm; ngưỡng ±0.05 không đổi):** với nêm, điểm "nền rắn sâu 150 µm" lấy tại **y = +200 µm** (rắn vì y ≥ 90 µm), kỳ vọng 1.0; điểm (y = 0, độ sâu 150 µm) chuyển thành điểm kiểm **bột dưới phần nhô**, kỳ vọng r. Vệt thường giữ nguyên. Ghi nhận: dưới tâm chùm, nêm y_a = 0 chỉ có bột dưới độ sâu 60 µm của lớp, và vì bao bể chảy chỉ sâu 98 µm nên vùng bột còn lại từ 98 µm trở xuống vẫn cách nhiệt; đây là đúng cơ chế cần thể hiện.

## 7. Kết quả S2.17 (19:57:45–20:03, 318 s, mã `a69c034`): **ĐẠT** cả chín mục; ghi lúc 20:03

`legacy/ckpt` không đổi. (a) tương đương: r = 1: sai 0 (phần dư PDE và `top_res`); e ≡ 0 và r = 0.02346: 9.3e-15 / 0 (≤ 1e-10). (b) `∇κ_s` autograd so với sai phân trung tâm: phân vị 99% sai số 2.2e-7 (≤ 1e-4; 1,393 thành phần không tầm thường). (c) `div(κ_s K ∇θ)` so với sai phân thông lượng trực tiếp: 5.2e-7 (≤ 1e-5). (d) bao bể chảy từ B3 (iter0): độ sâu tại y = 0 **90.0 µm** (cần 90.8 ± 3), rộng trên mặt **166.0 µm** (167 ± 4), sha `20034cb6`; bao từ M2″ (iter0): sâu 98.0 µm, nửa rộng 84.0 µm, sha `f2ecb131`. (e) bản đồ κ_s: vệt thường 1.000 / 0.023 / 0.023 / 1.000; nêm 1.000 / 0.023 / 0.023 / 1.000 (nền rắn y = +200 µm) / 0.023 (bột dưới phần nhô y = 0) / 0.023 / 1.000 (sai ≤ 0.05). (f) E_id của M2″: −4.61 mJ không mặt nạ; −4.61 mJ với mặt nạ r = 1 (lệch 0.00). (g) khói `runner.py B3W0i` 40 bước: thoát 0, tag `B3W0i_77e4b83d` (khác tag không băm), loss hữu hạn, có bao kế tiếp. (h) canary 500 bước Adam: **vệt thường**: tổng loss 9.11e-2 → 4.02e-2 (tỉ lệ 0.441 ≤ 0.7), 0.170 s/bước, E_res −0.6, E_id **−22.1** mJ, hụt T_min 0.86%, tỉ lệ năng lượng 0.769; **nêm** (từ M2″): 1.153e-1 → 5.25e-2 (0.455), 0.175 s/bước, E_res +5.5, E_id −10.0 mJ, hụt 0.86%, tỉ lệ 0.902. (i) hồi quy: tầng 2, S2.13, S2.16 đạt (114 s). Chấm dự đoán (mục 3): (a)–(g) đạt cả lượt đầu 70% → **sai một phần** (một lỗi đặc tả điểm kiểm ở (e) cho nêm, mục 6, mã đúng); (h) vệt thường đạt 75% → đúng; nêm đạt 55% → đúng; (i) 85% → đúng.

**Ghi nhận cho lượt đầy đủ (không phải tiêu chí của mốc test):** (1) loss khởi điểm khi bật mặt nạ (khởi động từ trọng số không cách nhiệt): [L_PDE, L_top] = [0.091, 7.4e-5] (vệt thường) và [0.115, 1.3e-4] (nêm), cỡ 100× giá trị hội tụ của B3 và M2″ (1e-3 đến 8e-3) nhưng thấp hơn nhiều so với 0.33 của B3 trên nêm lúc đầu (S1.2); không dùng liên tục hóa theo r. (2) Sau 500 bước, vệt thường có |E_res| nhỏ (−0.6) nhưng E_id **−22.1 mJ**: hai thước đo lệch nhau 21 mJ khi trường chưa hội tụ (số hạng cân bằng năng lượng huấn luyện E_res, không huấn luyện E_id); E_res ≈ E_id chỉ đúng với trường đủ hội tụ, nên quy tắc dừng cần cả hai (đúng như cổng đã đăng ký). (3) Chạy đầy đủ ngay theo mục 4.

## 8. Kết quả lượt đầy đủ F (20:04–20:21, 17 phút GPU, `experiments/f_insul.py`, `runs/F/B3i_it0/run.json`): vệt thường cách nhiệt vòng 0 **KHÔNG ĐẠT cổng**; theo mục 4 **dừng, báo cáo, không chạy lại, nêm chưa chạy**; ghi lúc 20:22

Chạy đúng đăng ký (`supervise`, không sập, `legacy/ckpt` không đổi). **Dùng hết 6,000 bước Adam, không có hai điểm kiểm liên tiếp đạt** (`iteration_budget`).

| | B3 (bột như đặc) | B3i vòng 0 (bột cách nhiệt) |
|---|---|---|
| E_res `probe` / E_id (mJ) | −7.4 / −5.6 | **−2.2 / −32.1** (a_res đạt, **a_id trượt**) |
| hụt T_min (ngưỡng 1%) | 0.67% | **1.00%** (trượt sát ngưỡng; T_min 245.8 K) |
| tỉ lệ (tích + bay hơi)/nạp | 0.805 | 0.726 |
| `check_physics` | 0 | **1 cảnh báo** (phép 1, hụt T_min) |
| L_PDE cuối | 1.05e-3 | **2.6e-2** (25 lần) |
| RMS vùng > 1000 K | 0.0398 | **0.2608** (6.5 lần) |
| bể chảy dài / rộng / sâu ở 70% (µm) | 517.7 / 167.2 / 90.8 | **293.1 / 138.9 / 87.6** |
| bao kế tiếp so với bao dùng | | độ sâu −4.0 µm, nửa rộng **−14.0 µm**, hiệu đối xứng 13.5% → **không hội tụ** |

Diễn biến: cổng đầy đủ ở mọi điểm kiểm: gate (a) trượt (a_id) từ bước 500 (E_id −22.6 → −34.0 → −32.1 mJ, không tiến); gate (b) 0.84–1.02%; loss L_PDE 3.8e-2 → 2.6e-2 (giảm 31% trong 6,000 bước). E_res nhỏ (−2.2 mJ) **nhưng là kết quả triệt tiêu**: chẩn đoán (2^18 điểm, `runs/F`): dải T ≤ 1000 K (96% thể tích) +29.6 mJ, dải 1000–T_sol −13.9 mJ, dải ≥ T_sol −11.6 mJ. E_id (không phụ thuộc đạo hàm bậc hai) vạch ra −32 mJ, tức 17% năng lượng nạp. Gate E_id đã làm đúng việc nó được thêm vào.

**Chẩn đoán vị trí sai số PDE (không huấn luyện, không phải mốc):** lớp chuyển tiếp κ_s ∈ (0.1, 0.9) chiếm **3.7%** thể tích nhưng **53.6%** loss (RMS 0.27, so với 0.05 ở vùng rắn và 0.043 ở bột cách nhiệt); theo vị trí so với laser: |dx| < 300 µm (24% thể tích) chiếm **81%** loss (RMS 0.12 và 0.16 so với 0.045 và 0.026 xa laser). Sai số tập trung ở nơi hệ số dẫn nhiệt đổi 43 lần trong vài chục µm, cạnh bể chảy, đúng chỗ gradient nhiệt độ lớn nhất.

**Chấm dự đoán (mục 4):** vệt thường cách nhiệt qua cổng 55% → **sai**; mặt nạ hội tụ trong ≤ 3 vòng, cho cả hai hình học (45%): không đánh giá được; ΔD cách nhiệt > +9.0 µm: **không đo được** (nêm chưa chạy, vệt thường không đạt). Thời gian dự báo 40–100 phút, thực tế 17 phút (dừng sớm theo quy tắc).

**Phạm vi của kết luận (số đo, không suy diễn).** Kết quả này là âm cho **công thức tối ưu hóa** (khởi động từ trọng số cũ, Adam 6,000 bước, mạng 6 × 24, độ mềm w_x = 8 µm), **không** là kết quả về vật lý của bột cách nhiệt: chiều dài bể 293 µm và rộng 139 µm của B3i không đáng tin vì trường không thỏa PDE (L_PDE 2.6e-2). Chưa kiểm được liệu khó khăn nằm ở bước nhảy r (43 lần), ở độ mềm của mặt nạ, hay ở sức chứa mạng.

## 9. Phân tích nguyên nhân sau lượt F (22:34; chẩn đoán không huấn luyện, **không đổi phán quyết ở mục 8**)

Trọng số B3i vòng 0 (`runs/F/B3i_it0`), 2^18 điểm đều (seed 777), mặt nạ vòng 0.

**1. Sai số nằm ở đâu (theo nguồn biến thiên của κ_s; ngưỡng |∇κ_s| > 0.005 /µm):**
| Vùng | thể tích | RMS phần dư | phần loss |
|---|---|---|---|
| không biến thiên κ_s | 92.6% | 0.047 | 39.8% |
| **mặt phân cách lớp bột tĩnh (z = 180 ± 25 µm)** | 6.4% | 0.173 | **37.4%** |
| **mép bao bể chảy (y, z), ngoài lớp z = 180** | 0.65% | **0.409** | 21.3% |
| mép trước mặt nạ (x), di chuyển | 0.29% | 0.153 | **1.3%** |
Hệ quả: **mặt nạ di chuyển không phải là khó khăn** (nỗi lo ban đầu về κ phụ thuộc t không có cơ sở). Khó khăn nằm ở các mép tĩnh theo (y, z), và còn 40% loss ở vùng trơn với RMS 0.047, gấp khoảng hai lần B3 (≈ 0.02): mạng chưa hội tụ nói chung.

**2. Điều kiện biên mặt trên** (L_top 2.3e-3, B3: 7.6e-5, 30 lần): 90.7% nằm ở vùng κ_s > 0.9 (trong bể, mặt trên nóng) và 52% ở |dx| < 100 µm; **không** do lớp bột cách nhiệt trên mặt (κ < 0.1: 3.8%). Nghĩa là điều kiện bay hơi ở mặt bể cũng bị ảnh hưởng, như một phần của việc nghiệm chưa hội tụ.

**3. Tốc độ giảm loss.** L_PDE theo bước Adam (500…5,500): 3.77e-2, 3.18e-2, 3.01e-2, 2.87e-2, 2.79e-2, 2.63e-2: giảm như bước^−0.15. Để về 1e-2 cần khoảng **650 lần** số bước hiện có: **Adam không thể đưa bài toán này về hội tụ**; đây là kẹt tối ưu hóa, không phải thiếu vài nghìn bước. Tiền lệ trong dự án: B3 sau Adam ở 8.5e-3 và **L-BFGS** đưa về 1.05e-3 (8 lần); M2 không cắt: L-BFGS 6.4e-3 → 4.3e-3. Công thức M2″ **bỏ L-BFGS** (vì số hạng cân bằng năng lượng và phạt dương lấy mẫu ngẫu nhiên mỗi bước, mục 17 của đăng ký 03, hạn chế 5); với vật lý cũ điều đó không hại vì M2″ khởi đầu thấp (1.4e-2 sau 500 bước) và đạt cổng ở 2,000 bước; với bột cách nhiệt, nó loại bỏ đúng công cụ đã làm nên bước tối ưu cuối.

**4. Sức chứa mạng không phải nghi phạm chính.** A.1 (bột cách nhiệt tĩnh, k_s/k_p = 42.6, mạng 6 × 24) từng đạt L_PDE/L_top cuối **2.63e-4 / 1.73e-5** sau 60 phút (Adam 10,000 + L-BFGS 15,000; `coding/PROGRESS.md` mục 4.1; vật lý đơn giản hơn: không ẩn nhiệt, bay hơi, k lỏng). Sự tương phản 43 lần ở mặt phân cách tĩnh biểu diễn được bằng mạng này; thất bại hiện nay là của công thức tối ưu.

**5. Mật độ điểm huấn luyện không phải nghi phạm.** Mật độ tương đối (tỉ lệ điểm / tỉ lệ thể tích) trên các vùng khó: mặt phân cách lớp bột **1.6×** (36,732 điểm), mép bao **5.0×** (11,915 điểm), mép trước mặt nạ **33×**. Các vùng khó đã được lấy mẫu dày; thêm điểm không phải lối thoát (khớp với kết quả âm của A1).

**6. Gate E_id đã phát hiện trường không đạt.** E_res −2.2 mJ là triệt tiêu giữa dải T ≤ 1000 K (+29.6 mJ) và dải nóng (−25.5 mJ); E_id −32.1 mJ (17% năng lượng nạp).

**Giả thuyết còn lại và cách kiểm (chưa chạy):** (H1) nhảy r = 1 → 0.02346 một lần quá lớn cho Adam; (H2) thiếu L-BFGS (không thể dùng cùng số hạng ngẫu nhiên); (H3) độ sắc của các mép: thang chuyển tiếp mặt phân cách lớp bột `iface_delta` 15 µm → tanh trên thang 3.75 µm; w_T = 50 K ↔ vài µm; trong khi cỡ hạt D50 = 30 µm (độ mềm cỡ 15–30 µm có cơ sở vật lý hơn). Mỗi giả thuyết cần một thí nghiệm có đăng ký riêng (05b).

## 10. Sửa đổi 05b (01:49, trước khi viết mã và chạy): hướng S3 — làm mềm mép hệ số dẫn nhiệt về cỡ hạt bột

Người dùng chọn S3 sau phân tích ở mục 9 (S1 + S2 là phương án tôi đề nghị; người dùng chọn S3, ghi lại đúng như vậy). **Đây là thay đổi mô hình, không chỉ thay đổi tối ưu hóa**: độ dày vùng chuyển tiếp của hệ số dẫn nhiệt trở thành một tham số vật lý của mô hình bột cách nhiệt. Lượt F ở mục 8 (mép sắc) giữ nguyên như bản ghi KHÔNG ĐẠT.

**Cơ sở vật lý.** Cỡ hạt D50 = 30 µm; mặt phân cách lớp bột/nền và biên vùng đã kết khối là các vùng nhám cỡ một đường kính hạt, không phải vài µm. Hiện tại: mặt phân cách lớp bột có độ rộng 10–90% khoảng 8 µm (tanh trên thang `iface_delta`/4 = 3.75 µm); mép bao bể chảy khoảng 3–5 µm (sigmoid theo nhiệt độ, w_T = 50 K); mép trước mặt nạ (w_x = 8 µm) đã rộng 35 µm.

**Thay đổi (chỉ cho hệ số dẫn nhiệt κ_s của bậc cách nhiệt `*s`; nhiệt dung C* dùng bản đồ vật liệu cũ, không đổi):**
1. Mặt phân cách lớp bột trong κ_s: tanh trên thang **a_I = 13.65 µm**, tức độ rộng 10–90% = 2.197·a_I = **30 µm** (= D50). Dùng `L.powder_index` với `dlt_s` thay thế (cùng hàm, kể cả nêm).
2. Mép bao bể chảy: thay sigmoid theo nhiệt độ bằng sigmoid theo **khoảng cách có dấu** tới đường bao T_max = T_sol (biến đổi khoảng cách Euclid trên lưới 2 µm): e = σ(−sd/w_e), **w_e = 6.83 µm**, tức độ rộng 10–90% = 4.394·w_e = **30 µm**, không phụ thuộc gradient nhiệt độ.
3. Mép trước mặt nạ giữ w_x = 8 µm (đã 35 µm).
Mọi tham số khác của mục 2 giữ nguyên (r = 0.02346, x_nose = 47 µm, lưới bao, công thức M2″, mục tiêu dừng, trần). Bậc mới `B3s` (vệt thường), `B3W0s` (nêm); bao vòng 0 dùng lại `runs/ENV/B3_iter0.pt` và `B3W0_iter0.pt` (bảng T_max không đổi, chỉ cách hiểu thay đổi).

**Mốc test S2.18 (chạy trước; trượt mục nào thì dừng, báo cáo, không chạy lượt đầy đủ):**
| ID | Kiểm | Tiêu chí |
|---|---|---|
| (a) | Độ rộng chuyển tiếp: s_k(z) dọc z tại (y = 0, ngoài bao, sau mũi) vệt thường; e(y) dọc y tại mặt trên từ bao B3 | mặt phân cách: độ rộng 10–90% = 30 ± 1.5 µm, tâm 180 ± 1 µm; mép bao: độ rộng 10–90% = 30 ± 2.5 µm, tâm tại nửa rộng bao ± 2 µm |
| (b) | Nhất quán bao: vùng e ≥ 0.5 có cùng độ sâu (tại y = 0) và nửa rộng trên mặt với `envelope_stats` (T_max ≥ T_sol) | lệch ≤ 2 µm mỗi đại lượng |
| (c) | Đạo hàm và divergence với cấu hình mềm, nêm, float64: như S2.17(b), (c) | phân vị 99%: ≤ 1e-4 và ≤ 1e-5 |
| (d) | Tương đương: bề rộng mặt phân cách bằng giá trị cũ (3.75 µm), e ≡ 0, r = 0.02346 so với `L.pde` với `kap_p = r`; và hai trường hợp r = 1, như S2.17(a) | sai ≤ 1e-10 |
| (e) | κ_s tại điểm cho trước (vệt thường, sau mũi, ngoài bao): z = 180 µm → 1 + (r − 1)·0.5; z = 210 µm → 1 + (r − 1)·s(+30 µm) = 0.0354; z = 150 µm → 0.988; tại mép bao (y = nửa rộng, z = 210 µm) → 0.52 ± 0.08 | sai ≤ 0.02 (các điểm mặt phân cách), ≤ 0.08 (mép bao) |
| (f) | Khói `runner.py run B3W0s` (Adam 40 bước, kiểm 20, `--min-steps 20`, thư mục tạm, bao từ M2″) | `run.json` DONE, tag khác tag của `B3W0i`, loss hữu hạn, có bao kế tiếp |
| (g) | Hồi quy: tầng 2 (S2.1–S2.4), S2.13, S2.17 chạy lại (cấu hình sắc mặc định không đổi) | đạt |
Trần GPU 15 phút (dừng ở 22.5). Không có canary tối ưu hóa riêng (lượt đầy đủ chỉ mất khoảng 17 phút cho vệt thường). Dự đoán: đạt cả bảy ngay lượt đầu 70%.

**Lượt đầy đủ F-mềm (nếu S2.18 đạt):** như mục 4 với `B3s` rồi `B3W0s` (vệt thường trước; vệt thường trượt thì dừng, không chạy nêm), cùng tiêu chí ĐẠT, cùng trần (6,000 bước Adam, hai điểm kiểm liên tiếp đạt sau 2,000 bước, `supervise`, tối đa 3 lần huấn luyện lại, tổng 120 phút), **cùng công thức tối ưu** (Adam, λ_E = 1, λ_P = 1000, khởi động từ B3 và M2″): một thay đổi duy nhất.

**Quy tắc quyết định sau lượt vệt thường vòng 0 (khóa trước khi chạy), so với B3i mép sắc (L_PDE cuối 2.6e-2, E_id −32.1 mJ):**
- **ĐẠT cổng** → tiếp tục theo quy trình.
- **Không đạt cổng nhưng cải thiện rõ** (L_PDE cuối ≤ 1.3e-2, tức ≥ 2 lần; và |E_id| ≤ 16 mJ): ghi "mép mềm giúp nhưng không đủ", **dừng**, đề xuất S1 (L-BFGS) trên cấu hình mềm; không tự chạy.
- **Không cải thiện rõ** (L_PDE cuối > 1.3e-2 hoặc |E_id| > 16 mJ): ghi "S3 không đủ", **dừng**, đề xuất S1 + S2.
- Cải thiện đo bằng chính chỉ số của B3i; chiều dài bể chảy chỉ ghi.

**Dự đoán (ghi trước, có thể sai).** Vệt thường vòng 0 qua cổng 30%; cải thiện ≥ 2 lần về L_PDE cuối (kể cả không qua cổng) 55%; nêm qua cổng, nếu vệt thường qua: 55%; mặt nạ hội tụ ≤ 3 lần huấn luyện lại, nếu cả hai qua cổng: 40%; ΔD cách nhiệt > +9.0 µm, nếu cả hai qua cổng: 85%. Lý do giữ thận trọng: 40% loss ở mục 9 nằm ở vùng trơn (RMS 0.047) và Adam kẹt theo bước^−0.15, hai điều S3 không trực tiếp xử lý.

**Giới hạn thêm (khóa).** Độ dày chuyển tiếp 30 µm là giả định cỡ hạt, không đo cho IN718 ở dự án này; kết quả cách nhiệt phụ thuộc nó. Nếu lượt đầy đủ qua cổng, độ nhạy theo độ dày (ví dụ 15 và 45 µm) là bước kiểm sau và hỏi người dùng; bản thân mô hình mềm không chứng minh bột cách nhiệt đúng.

## 11. Sửa đổi 05c (01:51, sau lượt chạy đầu của S2.18, trước khi chạy lại): lỗi đặc tả điểm kiểm (a) về tâm mép bao

Lượt đầu (01:50): độ rộng chuyển tiếp **đúng**: mặt phân cách lớp bột 30.00 µm (tâm 180.00 µm), mép bao 29.87 µm. Mục (a) trượt ở **tâm mép bao**: 85.0 µm so với "nửa rộng bao 83.0 ± 2". Nguyên nhân là **lỗi đặc tả của điểm kiểm**, không phải của mã: bao bể chảy trên mặt trải từ −82 đến +84 µm (lệch tâm +1 µm), nên "nửa rộng" (max − min)/2 = 83 µm không phải vị trí của mép phía +y; mép +y nằm giữa pixel nóng cuối (84 µm, sd = −2) và pixel lạnh đầu (86 µm, sd = +2), tức đúng ở **85.0 µm**, đúng như mã đo.

**Sửa (chỉ điểm kiểm (a); chặt hơn trước):** tâm mép bao phía +y so với **trung điểm giữa pixel nóng cuối và pixel lạnh đầu của bảng T_max trên hàng mặt trên**, dung sai ± 1 µm (thay vì so với nửa rộng trung bình, ±2 µm). Độ rộng 10–90% (30 ± 1.5 và 30 ± 2.5 µm) giữ nguyên. Điểm kiểm (b) và (e) dùng `envelope_stats` / nửa rộng và có cùng khả năng nhạy với độ lệch tâm 1 µm: (b) dung sai ± 2 µm vẫn đủ; (e) điểm mép bao có dung sai 0.08 trong κ_s (tương đương ± 2 µm ở độ dốc của mép), nên tôi giữ nguyên và ghi kết quả đo.

**Bổ sung 05c (01:51, sau lượt chạy thứ hai của S2.18): lỗi đặc tả thứ hai, ở điểm kiểm (e).** Lượt hai: (a), (b), (c) (S2.17b, c ở cấu hình mềm: 1.9e-8 và 2.3e-7), (d) **đạt**; (e) đạt ở ba điểm mặt phân cách (κ_s = 0.5117, 0.0354, 0.9881, trùng giá trị kỳ vọng) và **trượt ở điểm mép bao** (0.3486 so với 0.52 ± 0.08). Nguyên nhân là lỗi đặc tả: điểm tôi đặt ở (y = nửa rộng bao = 83 µm, **z = 210 µm**, tức độ sâu 30 µm), nhưng mặt cắt bể chảy là hình bán nguyệt nên ở độ sâu 30 µm đường bao hẹp hơn trên mặt; tại y = 83 µm điểm đó nằm ngoài bao khoảng 5 µm (e ≈ 0.33), và κ_s = 0.349 đúng với e = 0.33. **Sửa (chỉ điểm này):** điểm mép bao lấy ở (y = vị trí mép +y trên hàng mặt trên của bảng = 85.0 µm, z = Lz), kỳ vọng κ_s = 1 + (r − 1)·s(+60 µm)·0.5 = 0.5119, dung sai ± 0.05 (chặt hơn 0.08). Ba điểm mặt phân cách giữ nguyên.
