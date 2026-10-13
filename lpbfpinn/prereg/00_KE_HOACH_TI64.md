# Prereg: PINN nhiệt cho overhang không support, Ti-6Al-4V, LPBF

Trạng thái: **ĐÃ KHÓA (2026-10-10)** cho các giá trị ở mục 2. Chưa chạy huấn luyện nào.
Mọi thay đổi sau khi khóa phải ghi lại kèm ngày giờ, không sửa lặng lẽ.

## 1. Phạm vi

- Vật liệu: Ti-6Al-4V.
- 300 W, 0.8 m/s, r_b = 40 µm, lớp bột 60 µm, một đường quét.
- Hình học: overhang với mặt nghiêng 45° hướng xuống. Không mô phỏng vệt thường.
- Đầu ra: T(x,y,z,t) đầy đủ trên miền overhang. Chỉ báo là giá trị tuyệt đối tại vị trí cố định (mục 7).
- Không dự đoán ứng suất, biến dạng, hay võng (giai đoạn sau).
- Không hiệu chỉnh FEA trong suy luận (nhánh riêng sau này).
- Không dùng FD/FEA trong loss. Không dùng FD/FEA làm tham chiếu kiểm chứng; kiểm chứng dựa trên nghiệm giải tích, cân bằng năng lượng và hội tụ (mục 8).
- Bay hơi giữ (300 W, khôi phục bay hơi).
- DeepXDE làm thư viện nền. Code viết mới trong `lpbfpinn/`.
- Cấu hình bằng dataclass Python; mỗi lượt chạy có manifest JSON.

## 2. Giá trị vật liệu và quá trình (đã khóa)

| Tham số | Giá trị | Nguồn / ghi chú |
|---|---|---|
| T_sol | 1878 K | Cook & Ritchie 2023, bảng 2 |
| T_liq | 1928 K | Cook & Ritchie 2023, bảng 2 |
| ΔT = T_liq − T_sol | 50 K | Hệ quả |
| L_f | 2.90×10⁵ J/kg | Cook & Ritchie 2023, bảng 2 |
| T_b | 3315 K | Cook & Ritchie 2023, bảng 2 |
| T_amb (nhiệt độ môi trường và ban đầu) | 298 K | Khóa ngày 2026-10-10 theo phản hồi review PR #1. Dùng cho đáy, biên trên (đối lưu, bức xạ) và điều kiện ban đầu. |
| p_amb (áp suất khí quyển, bay hơi) | 101325 Pa | Khóa ngày 2026-10-10 theo phản hồi review PR #1. Áp suất khí quyển chuẩn. |
| L_v(0) | 9.83×10⁶ J/kg | Cook & Ritchie 2023, bảng 2 |
| L_v(T) | L_v(0)·√(1 − T²/T_c²), T_c = 7845 K | Cook & Ritchie 2023, phương trình 12 |
| M (khối lượng mol) | 0.0467 kg/mol | Cook & Ritchie 2023, bảng 2 (trung bình có trọng số hợp kim) |
| φ (hệ số bay hơi) | 0.82 | Cook & Ritchie 2023, phương trình 11 |
| h_c (đối lưu, biên trên) | 40 W/m²·K | Cook & Ritchie 2023, phương trình 14 |
| α_s (hấp thụ, chuẩn) | 0.27 | Cook & Ritchie 2023, tập chế độ dẫn nhiệt, 1.07 µm |
| φ_bột (độ rỗng) | 0.405 | Khóa ngày 2026-10-10 theo duyệt. Zhang et al. 2019 (Data in Brief 27:104559): p = 1 − 2.63/4.42 (mật độ bột/rắn Ti-6Al-4V, Gong et al.). Thay 0.60 cũ (nhầm phần đặc với độ rỗng). |
| d50 | 30 µm | Khóa ngày 2026-10-10 theo đề xuất đã được duyệt. Li et al. 2022 (Powder Technology 397:117012), bột Ti-6Al-4V 15–45 µm, d50 = 30 µm, là giả định phổ biến trong mô phỏng SLM. Không phải đo đạc trên bột cụ thể. |
| k_gas (argon) | 0.01772 W/m·K | Khóa ngày 2026-10-10 theo đề xuất đã được duyệt. Li et al. 2022, k của pha khí (argon) = 1.772×10⁻² W/m·K. Bài không ghi nhiệt độ áp dụng. |
| ε (bức xạ, biên trên) | 0.35 | Cập nhật ngày 2026-10-10: Masoomi et al. 2017 qua Zhang et al. 2019 (Data in Brief 27:104559), bề mặt lớp hoạt động. Độ nhạy 0.2–0.6. |
| f_L | 1 (chuẩn) | Độ nhạy 2.8 |

Các giá trị có nguồn cho k_s, c_p, ρ theo nhiệt độ: lấy từ Cook & Ritchie (bảng 2), nội suy tuyến tính giữa các điểm đã cho. Đây là phương pháp của bài này.

## 3. Hấp thụ laser

- Chuẩn: α_s = 0.27 (Cook & Ritchie). Bài này đo trong LPBF, 1.07 µm, bề mặt nóng chảy, chế độ dẫn nhiệt.
- Độ nhạy: α_s = 0.33 (giá trị khi tập toàn bộ thí nghiệm, Cook & Ritchie bảng 3).
- Lưu ý: giá trị 0.27 được hiệu chỉnh trong mô hình có phản xạ đa lần. Mô hình của chúng ta không có phản xạ đa lần. Đây là một khác biệt cần ghi trong luận văn.
- Tương quan Moser cho bột (α_eff, β, k_bột) tính từ α_s = 0.27: α_eff ≈ 0.355, 1/β ≈ 28.5 µm. Chưa tính lại k_bột với k_gas chưa có nguồn.

## 4. Bay hơi trên biên trên

Cook & Ritchie, phương trình 11:

```
q_evap(T) = φ · L_v(T) · √(M / (2πRT)) · p_amb · exp[ −(L_v(0)·M/R)·(1/T − 1/T_b) ]
L_v(T)    = L_v(0) · √(1 − T²/T_c²)
```

Lưu ý khác với bản trước: prefactor dùng L_v(T), số mũ dùng L_v(0); có thêm φ = 0.82.

Thông lượng so với đỉnh laser (α_s = 0.27, 3.22×10¹⁰ W/m²):

| T (K) | q_evap / q_đỉnh |
|---|---|
| 3000 | 0.22% |
| 3300 | 1.11% |
| 4000 | 17.9% |
| 5000 | 226% |

Quy tắc: báo cáo tỷ lệ bay hơi nếu nhiệt độ đỉnh vượt 4000 K. Nếu vượt 5000 K thì mô hình không hợp lệ (không có keyhole, không có áp suất giật).

## 5. Lớp chuyển pha và lịch sử pha

**Tỷ lệ lỏng tuyến tính** (Cook & Ritchie phương trình 5; quy tắc đòn bẩy của Moser):

```
g(T) = 0                 T < T_sol
g(T) = (T − T_sol)/ΔT    T_sol ≤ T ≤ T_liq
g(T) = 1                 T > T_liq
```

Nhiệt dung biểu kiến: c_eff(T) = c_p(T) + L_f·dg/dT, với dg/dT = 1/ΔT trong khoảng. Đỉnh biểu kiến ≈ 2.90×10⁵/50 = 5800 J/kg·K.

Hàm Gauss σ = 150 K của IN718 đã bỏ.

**Lịch sử pha M(x,t):** cực đại tích lũy theo thời gian của g(T), không giảm. Số vòng lặp ngoài là 4 (lựa chọn của code cũ, không có trong hai bài). Tiêu chí dừng: đọc lại `prereg/08` của nhánh cũ trước khi chạy và ghi lại vào đây.

## 6. Biên

- Điều kiện ban đầu: T(x, y, z, 0) = T_amb = 298 K trên toàn miền (cứng).
- Đáy: T = T_amb (cứng).
- Hai biên bên: cách nhiệt, ∂T/∂n = 0. Cần kiểm tra trong code.
- Biên trên:

```
−k·∂T/∂z = q_evap(T) + h_c·(T − T_amb) + ε·σ_SB·(T⁴ − T_amb⁴)
```

Với h_c = 40 W/m²·K (Cook & Ritchie: không ảnh hưởng đáng kể trong vùng nóng). Chạy độ nhạy ε ∈ {0.2, 0.3, 0.6} và h_c ∈ {20, 40}.

## 7. Chỉ báo overhang (đã duyệt ngày 2026-10-10)

Hình học (mục 12): phần rắn là tam giác phía bên phải trên; mép treo là đường chéo 45° từ (y=0, z=180 µm) đến (y=180 µm, z=0); bột nằm bên dưới đường chéo.

- **Điểm B (không hỗ trợ):** ngay dưới mép treo, gần đầu nhọn, y ≈ 30 µm, z ≈ 150 µm. Bột bên dưới, khối rắn phía trên mỏng, nhiệt khó thoát.
- **Điểm A (có hỗ trợ):** ngay dưới mép treo, gần chân rắn, y ≈ 150 µm, z ≈ 30 µm. Khối rắn phía trên dày, nhiệt có đường dẫn về đế.
- Đo tại thời điểm laser đi qua mỗi điểm (cùng x với tâm laser khi đó).

**Chỉ báo chính:** thời gian lưu trên T_sol tại B, t_B = thời gian từ lúc T vượt T_sol đến lúc T giảm dưới T_sol.
**Chênh lệch chính:** Δt = t_B − t_A.

**Chỉ báo phụ (mô tả):**
- Nhiệt độ đỉnh tại A và B. Nếu đỉnh vượt 4000 K, ghi chú rằng kết quả chỉ mang tính định hướng do bay hơi chi phối.
- Tốc độ nguội tại B qua khoảng T_sol → T_liq.

**Không có ngưỡng.** Chỉ báo chỉ mô tả định lượng và so sánh giữa các cấu hình.

**Báo cáo:**
- Giá trị trung bình và độ phân tán trên ít nhất 3 seed. Coi là ổn định nếu độ phân tán dưới 10% trung bình.
- Độ nhạy: thay α_s (0.27 và 0.33), h_c (20 và 40), φ và L_v. Kết luận về dấu và độ lớn của Δt được xem là đúng nếu thay đổi dưới 10%.

## 8. Lấy mẫu thích ứng và kiểm chứng

**Lấy mẫu thích ứng:** tăng mật độ điểm gần các đẳng nhiệt T_sol và T_liq, và gần đỉnh laser. Báo cáo tỷ lệ điểm trong dải chuyển pha.

**Stefan 1D (cổng bắt buộc trước bài chính):**
- Nghiệm giải tích (Neumann, Carslaw & Jaeger) chỉ có cho biên chảy sắc. Dải 50 K không có nghiệm đóng.
- Kiểm chứng: (a) so với nghiệm Neumann trong giới hạn ΔT → 0; (b) với ΔT = 50 K, kiểm tra hội tụ khi ΔT giảm dần và kiểm tra cân bằng năng lượng.
- Ngưỡng đạt: lấy từ `prereg/08`, khóa trước khi chạy.

**Các cổng khác:**
1. Cân bằng năng lượng trong mọi lượt chạy (nhiệt đưa vào = tích lũy + tổn thất).
2. Hội tụ theo số điểm lấy mẫu và số vòng lặp ngoài.
3. Độ nhạy: α_s (0.27, 0.33), f_L (1, 2.8), h_c, ε, φ, φ_bột.
4. Kiểm chứng không dùng FD/FEA: nghiệm giải tích khi có, cân bằng năng lượng, hội tụ theo lưới và số vòng. Kiểm chứng bằng nghiệm nhân tạo (MMS) đề xuất chờ duyệt, xem mục 13.

## 9. Việc còn mở (không chặn việc viết code)

1. d50 bột Ti-6Al-4V: cần nguồn.
2. k_gas argon: cần nguồn hoặc xác nhận.
3. ε: cần nguồn Ti, hoặc giữ như giả định và chạy độ nhạy.
4. Ngưỡng cổng và tiêu chí dừng vòng lặp ngoài: đọc `prereg/08`.
5. Chỉ báo overhang cụ thể (mục 7).

## 10. Nhật ký thay đổi

- 2026-10-12: cập nhật cơ sở FV theo Yang 2026 (mục 24): bỏ T_cap 6000 K (dùng T_c), α cơ sở 0.5 cho bột, k_ratio cơ sở 0.05, φ = 0.82 = 1 − β_R.
- 2026-10-12: chuyển bộ thông số chính sang Charles et al. 2022 (mục 22); lớp bột giữ Moser; giả định bay hơi Cook (mục 21).
- 2026-10-12: ghi giả định bay hơi (mục 21): A_accom của Charles không có giá trị trong bài, giữ công thức Cook với φ = 0.82.
- 2026-10-12: chọn 175 W, 0.39 m/s, 40 µm làm quá trình chính của bài overhang (mục 20), thay thí nghiệm 4 (mục 19).
- 2026-10-12: đổi mặc định quá trình của bài overhang sang thí nghiệm 4 (mục 19), sai lệch có chủ đích so với mục 2.
- 2026-10-12: ghi nhận kết quả PINN giai đoạn 0 theo thí nghiệm 4 (mục 18): tiêu chí năng lượng 5.3% không đạt, các tiêu chí khác đạt. Chuyển sang bộ giải số thể tích hữu hạn làm tham chiếu cho giai đoạn 0 (sửa mục 13 cho phạm vi này).
- 2026-10-12: thêm mục 17 chọn thí nghiệm 4 của Cook & Ritchie (120 W, 0.6 m/s, r_b 71 µm, α_s 0.27) làm kiểm chứng giai đoạn 0. Thay mốc 2500–3000 K tự đặt bằng mốc từ bài báo. Mục 2 (bài overhang chính) không đổi.
- 2026-10-10: khóa φ_bột = 0.405 (thay 0.60) và giữ h_c = 40 theo duyệt.
- 2026-10-10: khóa d50 = 30 µm và k_gas = 0.01772 W/m·K theo Li et al. 2022.
- 2026-10-10: khóa mục 2 theo quyết định của bạn. Thay α_s 0.35 → 0.27; T_sol/T_liq 1873/1923 → 1878/1928; L_v 9.0 MJ/kg hằng số → công thức Cook; M 0.04787 → 0.0467; φ_bay hơi 1 → 0.82; h_c 20 → 40.

### Đề xuất chờ duyệt (ngày 2026-10-10)

- ĐÃ DUYỆT 2026-10-10, khóa 0.405. **φ_bột (độ rỗng): đề xuất 0.405 thay cho 0.60.** Zhang et al. 2019 tính p = 1 − 2.63/4.42 = 0.405 từ mật độ bột và mật độ rắn của Ti-6Al-4V. Giá trị 0.60 trước đó là phần thể tích đặc (packing), bị nhầm với độ rỗng. Nguồn Springer 10.1007/s40516-022-00185-3 nói mật độ đóng gói khoảng 40%, mâu thuẫn với số đo này. Nếu đổi, hệ số nhiệt dung bột (1 − φ) đổi từ 0.40 sang 0.595.
- ĐÃ DUYỆT 2026-10-10, giữ 40. **h_c: đề xuất giữ 40 W/m²·K (Cook & Ritchie), kiểm tra độ nhạy với 12.7 W/m²·K (Masoomi et al. 2017 qua Zhang et al. 2019).** Hai giá trị này đến từ hai bộ mô hình khác nhau.

## 11. Tiêu chí cổng Stefan 1D (đã duyệt ngày 2026-10-10)

- Sai số tương đối vị trí mặt chảy so với nghiệm Neumann: dưới 5% tại mọi thời điểm kiểm tra (τ = 0.5, 1.0, 2.5, 5.0).
- Sai số L2 tương đối của T gần mặt chảy: dưới 3% tại mọi thời điểm kiểm tra.
- Kết quả lượt 20000 vòng (`runs/gate_stefan/summary.json`, trên đám mây, CPU, lấy mẫu đồng đều): sai số mặt chảy 12.2%, 4.9%, 3.5%, 7.1%; sai số T gần mặt chảy 1.8%, 1.0%, 1.0%, 1.7%. **Chưa đạt** tiêu chí mặt chảy tại τ = 0.5 và τ = 5.0. Tiêu chí T đạt tại mọi thời điểm.

## 12. Hình học nêm (đã xác nhận ngày 2026-10-10)

- Nguồn hình: Figure 4 (hình học overhang đang được L-PBF, 3D view và right view) trong bài bạn cung cấp.
- Chiều dài 1500 µm theo x (hướng quét). Chiều rộng 380 µm theo y. Phần rắn cao 180 µm, bột phủ 60 µm, tổng cao 240 µm theo z.
- Mặt treo là mặt chéo 45° của phần rắn, hướng về phía bột. Bột nằm phía trên mặt treo, rắn nằm phía dưới và bên phải.
- Lưu ý: chiều rộng 380 µm thay cho 480 µm của cấu hình cũ.
- Nguồn bài báo: Figure 4 (3D view và right view) của bài báo overhang L-PBF (bản đọc từ hình, không ghi thêm số liệu khác).
- Cập nhật ngày 2026-10-12: đầu nhọn phần treo (tip) đặt tại y_tip = 110 µm (đọc từ Figure 4b, sai số ±10 µm); đường chéo z + (y − y_tip) = 180 µm; tâm đường quét tại y = y_tip, z = 240 µm (mặt bột). Chùm 40 µm nằm trọn trong miền 0–380 µm. Trước đó code đặt tâm tại y = 0 (nửa chùm ngoài miền), đã sửa.

## 13. Cập nhật ngày 2026-10-10: bỏ FD làm tham chiếu kiểm chứng

- Thay đổi: không dùng FD/FEA làm tham chiếu kiểm chứng, cũng không dùng trong loss. Mục 1 và mục 8 đã cập nhật.
- Hệ quả đối với kế hoạch:
  - Bài overhang không có nghiệm giải tích, nên kiểm chứng chỉ còn: cân bằng năng lượng toàn miền, hội tụ theo lưới và số điểm lấy mẫu, độ nhạy tham số, và đối chiếu định tính với vật lý (ví dụ hướng và độ lớn của nhiệt độ đỉnh, tốc độ nguội).
  - Cổng Stefan 1D không đổi, vì dùng nghiệm Neumann giải tích, không dùng FD.
  - **Đã duyệt ngày 2026-10-10:** kiểm chứng bằng nghiệm nhân tạo (method of manufactured solutions) cho phương trình 3D với tính chất hằng số, để kiểm tra residual và điều kiện biên mà không cần FD.
  - **Đã duyệt ngày 2026-10-10:** bài overhang không còn tham chiếu số; cổng Stefan không đổi.
  - Chưa chốt: cách đánh giá chỉ báo overhang (mục 7).

- 2026-10-10: duyệt chỉ báo overhang (mục 7): điểm A và B theo hình học mục 12, chỉ báo chính là t_B(T_sol) và Δt = t_B − t_A.

## 14. Kết quả cổng Stefan 1D: ghi nhận, chưa khóa là đạt (ngày 2026-10-10)

Quyết định của người nghiên cứu: giữ nguyên tiêu chí mục 11 (mặt chảy dưới 5%, nhiệt gần mặt chảy dưới 3%, mọi thời điểm) và ghi nhận kết quả trung thực, không khóa cổng là đạt.

**Lịch sử điều chỉnh** (chạy trên GPU RTX 4080, PyTorch 2.14.1, 20000 vòng Adam, 4000 điểm miền, 643 giây một lượt):
1. Lấy mẫu đồng đều, nhiệt ẩn tuyến tính, không cứng hóa biên: seed 0 đạt, seed 1 và 2 không đạt (ở mặt chảy).
2. Thêm L-BFGS sau Adam: không cải thiện đáng kể.
3. Lấy mẫu thích ứng gần mặt chảy: xấu hơn ở mọi seed.
4. Nhiệt ẩn trơn (smoothstep): xấu hơn, bị loại.
5. Cứng hóa biên x = 0 và x = L: cải thiện thời điểm muộn, thời điểm τ = 0.5 vẫn khoảng 10–13%.
6. Cứng hóa biên và lấy mẫu sớm trong τ ≤ 1: seed 1 cải thiện mạnh, seed 0 và 2 vẫn sai ở τ = 0.5.
7. Cứng hóa biên và lấy mẫu sớm trong τ ≤ 0.5 (cấu hình hiện tại, "tref05"): seed 0, 1, 2, 3 đạt; **seed 4 không đạt** ở τ = 0.5 (5.10%).
8. Seed 4 với giảm tốc độ học (5000 vòng, giảm một nửa): τ = 0.5 đạt (3.95%), nhưng **τ = 5.0 không đạt** (5.22%).

**Kết quả hiện tại:**
- Cấu hình tref05 (không giảm tốc độ học): 4/5 seed đạt đầy đủ (seed 0, 1, 2, 3), seed 4 không đạt do vượt 5% ở τ = 0.5 với biên 0.1 điểm phần trăm.
- Sai số không đạt quanh ngưỡng 5%, cùng cỡ với dao động giữa các seed. Đây là dấu hiệu của dao động ngẫu nhiên, không phải lỗi hệ thống có thể sửa dễ dàng bằng một tham số.

**Ý nghĩa:**
- Cổng Stefan chưa đạt ổn định theo tiêu chí đã duyệt. Không khóa là đạt.
- Phương pháp PINN cho bài toán một chiều đã cho sai số nhỏ (dưới 3% với nhiệt độ, dưới 5% với mặt chảy trong hầu hết các trường hợp). Đây là bằng chứng trung thực cho phần kiểm chứng trong luận văn.
- Bài overhang chưa được viết. Kết quả này là giới hạn cần nêu khi trình bày phương pháp.

**Cấu hình để tham chiếu:** biên cứng tại x = 0 và x = L, điều kiện ban đầu dạng mềm, lấy mẫu sớm 50% trong τ ≤ 0.5, phần còn lại đồng đều, Adam 20000 vòng, 4000 điểm miền, nhiệt ẩn tuyến tính. Kết quả đầy đủ nằm trong `results/` trên GitHub.

## 15. Ngưỡng chấp nhận kiểm chứng 3D bằng nghiệm nhân tạo (đã duyệt ngày 2026-10-10, trước khi chạy)

- Cấu hình: `lpbfpinn/mms3d.py`, cấu hình tref05 (biên và điều kiện ban đầu cứng hóa, lấy mẫu sớm 50% trong τ ≤ 0.5), Adam 20000 vòng, 20000 điểm miền, seed 0.
- Tiêu chí: sai số tương đối L2 so với nghiệm nhân tạo dưới **2%** tại mọi thời điểm kiểm tra (t = 0.05, 0.25, 0.5, 1.0).
- Lập luận: ngưỡng này đủ nhỏ để sai số nhiệt độ không làm sai thời gian lưu t_B quá 10% (ngưỡng độ nhạy của mục 7), và nằm trong mức dao động đã thấy ở cổng Stefan.
- Nếu không đạt: không kết luận đường ống 3D đúng. Phải chẩn đoán (số vòng, số điểm, lỗi residual) trước khi chạy bài overhang. Không đổi ngưỡng sau khi có kết quả.

- 2026-10-10 (sau duyệt mục 7): dời điểm đo ra ngoài vùng chuyển tiếp 30 µm. B: y = 30 µm, z = 125 µm (25 µm dưới mặt treo). A: y = 150 µm, z = 5 µm (25 µm dưới mặt treo, cách đáy 5 µm). Lý do: tại 5 µm dưới mặt treo chỉ số bột ≈ 0.6, chưa thuần.

## 16. Lượt chính thức đầu tiên của bài overhang (ngày 2026-10-10)

- Seed 0, 20000 bước Adam tổng, 4 vòng lặp ngoài (5000 bước mỗi vòng), 20000 điểm miền, 50% lấy mẫu tập trung quanh tâm laser, lưới M 20 µm (x, y), 10 µm (z), 30 mốc thời gian.
- Điều kiện dừng vòng lặp ngoài: trung bình |ΔM| < 0.05 **và** max M > 0 (đã có ô nóng chảy). Sửa từ điều kiện chỉ dựa trên ΔM, vì điều kiện cũ có thể dừng sớm khi chưa có vùng nóng chảy.
- Quy tắc báo cáo: chỉ báo t_B và Δt chỉ có ý nghĩa nếu B vượt T_sol trong đường quét. Nếu không, báo cáo "không đạt đến nóng chảy" và không suy ra t_B.
- Tiêu chí hội tụ loss PDE: đề xuất ghi lại sau khi có kết quả; không đặt ngưỡng trước khi chạy.

## 17. Kiểm chứng giai đoạn 0 theo thí nghiệm 4 của Cook & Ritchie (ngày 2026-10-12, đề xuất chờ chạy)

Phạm vi: chỉ áp dụng cho bài kiểm chứng giai đoạn 0 (Ti hằng số, không nhiệt ẩn, không bột, không bay hơi). Không đổi mục 2 (quá trình khóa 300 W, 0.8 m/s, 40 µm) cho bài overhang chính; bài overhang vẫn chờ quyết định riêng.

- Nguồn: Cook & Ritchie 2023, *Optics & Laser Technology* 162:109247, thí nghiệm 4 (bảng 1).
- Quá trình: P = 120 W, v = 0.6 m/s.
- Bán kính chùm: r_b = 71 µm, lấy tại bề mặt tấm (mục 3.3). Bài báo cho bán kính thay đổi theo độ cao (71 µm → 30 µm tại mặt tiêu cách 2.5 mm); mô hình dùng một r_b hằng số tại mặt trên, đây là điểm xấp xỉ cần ghi trong luận văn.
- Hấp thụ: α_s = 0.27 (mục 3), α_eff ≈ 0.355 theo Moser. Cần ghi: bài báo hiệu chỉnh 0.27 trong mô hình có phản xạ đa lần, mô hình này không có.
- Vật liệu: Ti hằng số theo mục 11 (ρ = 4106, c = 838, k = 33.1). Bài báo dùng tính chất phụ thuộc nhiệt độ; khác biệt này làm cho so sánh chỉ mang tính định tính về nhiệt độ đỉnh.
- Mốc tham chiếu từ bài báo (đọc từ hình, không phải số liệu bảng):
  - Vùng nóng chảy: rộng 126.1 µm, sâu 42.7 µm (bảng 1, thí nghiệm 4). So sánh với đường đẳng nhiệt 1928 K của mô phỏng nhiệt độ cực đại.
  - HAZ: rộng 156.0 µm, sâu 57.8 µm (bảng 1). So sánh với đường đẳng nhiệt 1280 K.
  - Nhiệt độ đỉnh: khoảng 2000–2600 K (ước lượng từ hình 6 và 9b, sai số ±300 K). Không dùng mốc 2500–3000 K tự đặt trước đây.

Tiêu chí chấp nhận (đề xuất, chưa chạy; không đổi sau khi có kết quả):
1. Nhiệt độ đỉnh trong khoảng 2000–2600 K tại tâm vết laser.
2. Chiều rộng và chiều sâu vùng 1928 K, và vùng 1280 K, sai lệch không quá 20% so với bảng 1.
3. Sai số cân bằng năng lượng |rel_error| < 5%.
4. Không có nhiệt độ dưới 298 K ngoài sai số số học (cho phép tối đa 1 K).

Không đạt tiêu chí 1 hoặc 3 thì không kết luận pipeline đúng; phải chẩn đoán trước khi chuyển sang bài overhang.

- Cấu hình thực hiện: `LPBF_PROC=exp4` (cần thêm vào `config.py`, chưa làm). Cần sửa thêm: vận tốc trong T_END (hiện cố định 0.8 m/s) và α (hiện lấy từ Moser). Hai thay đổi này chưa làm, chờ duyệt.

## 18. Kết quả PINN giai đoạn 0 theo thí nghiệm 4 và chuyển sang bộ giải số (ngày 2026-10-12)

Kết quả ghi nhận (không đổi tiêu chí mục 17, không đổi ngưỡng sau khi có kết quả):

| Lượt | Cấu hình | Nhiệt độ đỉnh (x = 0) | Sai số năng lượng | Residual PDE (RMS/RMS θ_t) |
|---|---|---|---|---|
| exp4_volume_10k | Adam 10000, nguồn thể tích, nonneg | 2339 K | +0.362 | 0.46–0.51 |
| exp4_volume_lbfgs | + L-BFGS | 2467 K | +0.066 | 0.02–0.04 |
| exp4_refine_lbfgs | + Adam tinh chỉnh 5000 trên 20000 điểm, + L-BFGS | 2466 K | +0.053 | 0.02–0.04 |

- Tiêu chí 1 (đỉnh 2000–2600 K): **đạt**.
- Tiêu chí 4 (không có T < 298 K): **đạt**.
- Tiêu chí 3 (|sai số năng lượng| < 5%): **không đạt**, sai số 5.3%, sát ngưỡng.
- Tiêu chí 2 (kích thước vùng nóng chảy): không đánh giá được, vì hình học hộp overhang (y_tip = 110 µm) không khớp thí nghiệm 4 (tấm phẳng, laser trên mặt đối xứng y = 0).

Kết luận: PINN giai đoạn 0 chỉ đạt gần đúng với cấu hình hiện tại. Residual giảm mạnh nhờ L-BFGS nhưng không giảm thêm khi tăng điểm lấy mẫu và tinh chỉnh; sai số năng lượng còn lại có dấu nhất quán (mạng thiếu năng lượng), nên không coi là nhiễu thống kê. Các lỗi đã sửa trong quá trình (không ảnh hưởng kết luận): nguồn laser bị tính hai lần ở bản nondim; diện tích lưới trong đoạn tính năng lượng (chỉ ảnh hưởng thông lượng đều, đã kiểm chứng bằng nghiệm giải tích); nhiệt độ dưới môi trường (đã thêm ràng buộc θ ≥ 0).

Hướng tiếp theo, theo quyết định của người nghiên cứu (2026-10-12):
- Viết bộ giải số thể tích hữu hạn cho giai đoạn 0 (cùng cấu hình exp4, tính chất Ti hằng số, nguồn thể tích Beer–Lambert, biên như mục 6), bảo toàn năng lượng theo cấu trúc.
- Bộ giải số này là tham chiếu số cho giai đoạn 0. Kiểm chứng bằng nghiệm giải tích (cùng nghiệm θ = c(t + z²/2) dùng cho đoạn tính năng lượng) trước khi so với PINN.
- PINN giữ làm đối chứng; so sánh nhiệt độ đỉnh và sai số năng lượng với bộ giải số.
- Mục 13 được bổ sung: bộ giải số thể tích hữu hạn được phép làm tham chiếu cho giai đoạn 0 và thí nghiệm 4 (không dùng trong loss của PINN). Lý do: PINN không đạt tiêu chí năng lượng sau nhiều cấu hình; bộ giải số bảo toàn năng lượng theo cấu trúc, nên là tham chiếu độc lập. Bài overhang chính vẫn không có tham chiếu số (mục 13 không đổi cho bài này).

## 19. Sai lệch có chủ đích: thông số quá trình của bài overhang chính (ngày 2026-10-12)

- Thay mục 2 (300 W, 0.8 m/s, r_b 40 µm) bằng thí nghiệm 4 của Cook & Ritchie 2023 (120 W, 0.6 m/s, r_b 71 µm, bảng 1 và mục 3.3). Lý do: thí nghiệm 4 có dữ liệu kích thước vùng nóng chảy và HAZ để đối chiếu, và đã dùng làm kiểm chứng giai đoạn 0 (mục 17–18).
- Hệ quả đã biết: với 120 W, đỉnh nhiệt độ dự kiến khoảng 2000–2600 K, có thể không đạt T_sol (1878 K) trên toàn đường quét ở hộp overhang. Theo mục 16, khi không đạt nóng chảy thì báo cáo "không đạt đến nóng chảy" và không suy ra t_B.
- Cấu hình 300 W / 0.8 m/s / 40 µm vẫn có thể gọi bằng LPBF_PROC=prereg300 để so sánh; các kết quả cũ thuộc cấu hình đó.
- Mục 2 được giữ nguyên làm ghi chép lịch sử. Các thông số khác (vật liệu, hấp thụ, biên, hình học overhang) không đổi.
- Luận văn cần ghi sai lệch này và lý do; so sánh với cấu hình 300 W chỉ là tham khảo.

## 20. Chọn quá trình chính theo hiện tượng dross (ngày 2026-10-12)

- Quyết định của người nghiên cứu: trường hợp chính 175 W, 0.39 m/s (năng lượng cao, P/v ≈ 0.45 J/mm), r_b = 40 µm. Thay mục 19 (120 W, 0.6 m/s, 71 µm) làm mặc định của bài overhang.
- Lý do: hiện tượng cần giải thích (dross) xuất hiện ở chế độ keyhole, khi melt pool đâm xuống vùng bột. Thí nghiệm 4 (P/v ≈ 0.2 J/mm) nằm ngoài chế độ này, nên không tái hiện được cơ chế đó.
- Nguồn: Charles et al. 2022, *Additive Manufacturing* 50:102551, mục 2.4 và kết quả Hình 5–7.
- Bán kính chùm: giữ 40 µm theo mục 2. Bài báo không ghi giá trị số, nên đây là điểm chưa có nguồn; cần ghi trong luận văn.
- Hình học khác bài báo: bề mặt treo 45° của hộp hiện tại (mục 12), so với 35° trong bài báo; miền và độ dày bột cũng khác.
- Trường hợp đối chứng 90 W, 1.235 m/s (P/v ≈ 0.073 J/mm, không đâm xuống bột trong bài báo): đã chọn ngày 2026-10-12, gọi bằng LPBF_PROC=dross_low, r_b = 40 µm.
- Cấu hình cũ vẫn gọi được: LPBF_PROC=exp4 (kiểm chứng giai đoạn 0), LPBF_PROC=prereg300 (mục 2).

## 21. Giả định bay hơi cho bộ giải FV (ngày 2026-10-12)

- Charles et al. 2022 (eq. 11) dùng công thức bay hơi có hệ số thích nghi A_accom, với ΔH_lv = 9.7 MJ/kg là hằng số (Bảng 1). Bài **không cho giá trị A_accom**.
- Giả định: giữ công thức bay hơi của Cook & Ritchie (mục 4, φ = 0.82, L_v(T) phụ thuộc nhiệt độ, kẹp T_cap = 6000 K), không dùng A_accom.
- Lý do: công thức Cook đã khóa trong kế hoạch và có hệ số φ có nguồn; không có cơ sở chọn giá trị A_accom.
- Hệ quả: so sánh với Charles chỉ ở mức định tính cho phần bay hơi; sai lệch do giả định này phải được ghi trong luận văn.
- Cần kiểm chứng: độ nhạy theo φ (hoặc A_accom tương đương) trước khi kết luận về nhiệt độ mặt trên.

## 22. Chuyển bộ thông số chính sang Charles et al. 2022 (ngày 2026-10-12, quyết định của người nghiên cứu)

Thay mục 2 cho bài overhang và bộ giải FV chính. Mục 2 giữ nguyên làm ghi chép lịch sử; mục 17–18 (kiểm chứng PINN giai đoạn 0) vẫn dùng bộ Cook & Ritchie vì đã được kiểm chứng theo bộ đó.

Bộ thông số (Charles et al. 2022, Bảng 1 và eq. 5–15):
- ρ = 4400 kg/m³; c_p,s = 573, c_p,l = 750 J/kg·K; k_s = 13, k_l = 33 W/m·K; L_f = 286 kJ/kg; T_s = 1878 K, T_l = 1928 K; T_boil = 3315 K; ΔH_lv = 9.7 MJ/kg (hằng số).
- Pha: f_l tuyến tính trên [T_s, T_l]; k = (1 − f_l)k_s + f_l k_l (eq. 14); c_p theo eq. 15; enthalpy eq. 6.
- Biên trên: đối lưu h = 25 W/m²·K, bức xạ ε = 0.2 (eq. 9–10), T_surr = 300 K.
- Laser: thông lượng bề mặt Gaussian, eq. 12, không có nguồn thể tích.
- Quá trình: 175 W, 0.39 m/s, r_b = 40 µm (mục 20).

Giả định (không có trong bài, ghi rõ):
1. Hệ số hấp thụ laser A = 0.27 (giá trị hiệu chỉnh của Cook & Ritchie, không có trong Charles). Cần chạy độ nhạy.
2. Bay hơi dùng công thức Cook với φ = 0.82 và kẹp T_cap = 6000 K (mục 21), vì A_accom của Charles không có giá trị.
3. Lớp bột dùng Moser (k_ratio = 0.012, độ rỗng φ = 0.405), vì Charles không cho độ dẫn nhiệt của bột (họ dùng DEM). Đây là giả định mạnh nhất về lớp bột.
4. Bán kính chùm 40 µm (Charles không cho R_b).
5. Hình học hộp 45° (mục 12), khác bài báo (35°).
6. Đáy T = T_surr, mặt bên cách nhiệt.

Quy tắc báo cáo không đổi (mục 16): nếu B không vượt T_s thì báo cáo "không đạt đến nóng chảy".

Chưa thực hiện: hội tụ theo lưới (chỉ có 10 µm), độ nhạy theo A và φ, bộ thông số lỏng đầy đủ trong bài overhang (đã có trong mã).

## 23. Chốt kết quả đến ngày 2026-10-12 (phạm vi kết luận)

**Kết luận được phép nêu**
- Với hình học kế hoạch (hộp overhang, đường quét dọc mép tại y_tip = 110 µm), mô hình dẫn nhiệt của Charles (laser bề mặt, Bảng 1, bột Moser), 175 W / 0.39 m/s / r_b = 40 µm: điểm B (y = 30, z = 125 µm, sâu 115 µm dưới mặt) không đạt T_sol = 1878 K trong mọi trường hợp đã quét (φ 0–0.82, k_ratio 0.012–0.07, α 0.27–0.6, r_b 25–71 µm). Nhiệt độ lớn nhất tại B là 330 K trong dải này.
- Điểm A (y = 150, z = 5 µm, sát đế) không nóng lên (300–318 K) trong mọi lượt; A không cho thông tin về nóng chảy.
- Mô hình dẫn nhiệt không đưa vùng bột dưới overhang đến nóng chảy trong dải tham số đã quét. Đây là kết quả của mô hình, không phải dự đoán vật lý.

**Kiểm chứng đã đạt**
- Cân bằng năng lượng đúng đến sai số máy (≤ 2.6·10⁻¹³) trong mọi lượt FV.
- Nhiệt độ mặt hội tụ theo lưới: 4392 K (10 µm) và 4399 K (5 µm), chênh 0.16%.
- Công thức Moser (eq. 10–12) trong mã khớp với bài Moser 2019. Giá trị k_ratio 0.012 nhất quán với giá trị Moser cho 316 (≈1.6%).
- Công thức bay hơi của Cook (eq. 11–12) khớp với mã.

**Chưa đạt hoặc không kết luận được**
- Tiêu chí kiểm chứng giai đoạn 0 bằng PINN (mục 17): đạt nhiệt độ đỉnh và T_min, không đạt cân bằng năng lượng (5.3%). PINN không dùng cho kết luận chính.
- Nhiệt độ mặt trên (4300–5900 K trong các lượt có φ > 0; 34 500 K khi φ = 0) vượt điểm sôi 3315 K. Không báo cáo như giá trị dự đoán, chỉ nêu xu hướng.
- Áp suất phản lực tại mặt (Cook eq. 10) lớn hơn áp suất mao dẫn 13–45 lần tại 4000–4392 K. Mô hình không tính được hệ quả này, nên vùng mặt không đáng tin về hình dạng.
- Ô dưới 15 µm chưa hội tụ (chênh 7% giữa 10 và 5 µm).

**Giả định không có nguồn đầy đủ**
- Hệ số hấp thụ 0.27 (Cook, hiệu chỉnh cho dẫn nhiệt). Charles không cho số.
- Hệ số bay hơi φ = 0.82 (Cook, nêu là cận trên); A_accom của Charles không có giá trị.
- Độ dẫn nhiệt bột theo Moser, chưa kiểm chứng cho Ti (đo cho Ti64 khoảng 2.4–7%).
- r_b = 40 µm (kế hoạch, không có trong Charles).
- Kẹp T_cap = 6000 K (quy ước của chúng ta, không có nguồn).
- Hình học 45° (kế hoạch), bài Charles dùng 35°.

**Không được kết luận từ kết quả này**
- Cơ chế dross hoặc đâm sâu của melt pool (cần áp suất phản lực, Marangoni, dòng chảy; mô hình không có).
- Thời điểm t_B hoặc Δt (B không đạt T_sol nên không suy ra được).
- Kích thước vùng nóng chảy so với Charles hay Cook.

**Hướng tiếp theo đã dừng:** mô hình VOF với dòng chảy không khả thi trong điều kiện hiện tại (không có công cụ CFD, tài nguyên và thời gian hạn chế). Nếu cần làm sau, phải có A_p, B_p từ Lee & Zhang 2016 và quyết định dùng OpenFOAM hay tự viết.

## 24. Cập nhật cơ sở mô hình FV theo Yang et al. 2026 (ngày 2026-10-12)

Thay đổi so với mục 22–23 (quyết định của người nghiên cứu, dựa trên Yang et al. 2026, IJHMT 265:128801):

1. **Kẹp nhiệt độ bay hơi:** bỏ T_cap = 6000 K. Cơ sở dùng T_cap = T_c = 7845 K, vì L_v(T) = L_v0·√(1 − (T/T_c)²) bằng 0 tại T_c nên thông lượng bay hơi tự triệt tiêu. Yang không kẹp nhiệt độ.
2. **Hệ số hấp thụ:** cơ sở α = 0.5 (hiệu dụng cho bột, Yang §2.5.1: khoảng 0.5 dọc đường quét). Giữ α = 0.27 (Cook, bề mặt phẳng) làm cận dưới.
3. **Độ dẫn nhiệt bột:** cơ sở k_ratio = 0.05, nằm trong dải đo cho Ti64 (2.4–7%: Bartsch 2022, NIST). Moser (k_ratio = 0.012) giữ làm cận dưới. Lý do: Moser được kiểm chứng cho thép 316, số đo cho Ti cao hơn.
4. **Hệ số bay hơi:** φ = 0.82 (Cook) tương đương hệ số Hertz–Langmuir (1 − β_R) với β_R = 0.18 (Yang eq. 15). Điều này cung cấp nguồn cho A_accom = 0.82 (mục 21).

Chưa thay đổi (còn ghi là giả định hoặc quét):
- Đáy: Dirichlet T = T_surr (cơ sở). Yang dùng đáy cách nhiệt; đáy cách nhiệt là trường hợp quét sẽ bổ sung.
- Hình học 45° và r_b = 40 µm theo kế hoạch mục 2 và 20.
- Áp suất phản lực, Marangoni, dòng chảy: không mô hình hóa (mục 23).

Kết quả so sánh sẽ ghi vào `results/fv_v24/` trước khi kết luận.

**Kết quả độ nhạy điều kiện đáy (mục 24, GPU, 175 W, lưới 10 µm, cơ sở mới):** đáy Dirichlet 300 K và đáy cách nhiệt cho cùng kết quả: mặt trên 4636 K, dưới 15 µm 2770 K, B 311.8 K, A 300.1–300.2 K. Cân bằng năng lượng ≤ 1.4·10⁻¹³. Kết luận: điều kiện đáy không ảnh hưởng đến kết quả trong miền và thời gian quét đã dùng. Kết quả lưu tại `results/fv_bottom/`.

## 25. Chuyển sang PINN surrogate hybrid cho overhang (ngày 2026-10-12, quyết định của người nghiên cứu)

1. **Mục tiêu PINN:** dùng PINN làm surrogate hybrid (nhãn từ bộ giải FV đã kiểm chứng + ràng buộc PDE), thay cho PINN thuần. Lý do: PINN thuần chưa đạt cân bằng năng lượng ở giai đoạn 0 (mục 18), còn FV đã kiểm chứng.
2. **Tiêu chí chấp nhận surrogate:** sai số so với FV ≤ 2% tại điểm B và tại mặt trên (cùng lưới, cùng cấu hình), và sai số cân bằng năng lượng của PINN ≤ 5% (đo bằng cùng công cụ đã kiểm chứng). Tiêu chí này ghi trước khi huấn luyện; không đổi sau khi có kết quả.
3. **Keyhole và dross: ngoài phạm vi (quyết định của người nghiên cứu, 2026-10-12, phương án A).** Mô hình dẫn nhiệt và surrogate không mô phỏng keyhole, áp suất phản lực làm lõm mặt, Marangoni hay dòng chảy. Luận văn nêu rõ đây là giới hạn của mô hình và là hướng nghiên cứu sau này (VOF). Kết luận về nóng chảy tại B chỉ áp dụng trong mô hình dẫn nhiệt. Phương án B (giới hạn trên bán định lượng) và C (VOF) không thực hiện trong luận văn này trừ khi có quyết định khác bằng văn bản.
   - Điều kiện: người hướng dẫn chấp nhận phạm vi này (chưa xác nhận trong hội thoại).

## 26. Thiết kế surrogate PINN hybrid (ngày 2026-10-12, bản thiết kế, chưa huấn luyện)

**26.1 Mục tiêu và phạm vi.** Surrogate học T(x, y, z, t) từ nghiệm FV (mục 22–24) và ràng buộc PDE, để đánh giá nhanh trường nhiệt theo tham số. Không mô phỏng keyhole, dòng chảy, hay dross (mục 25.3).

**26.2 Tham số đầu vào.** Phương án đề xuất: giữ φ = 0.82, r_b = 40 µm, v = 0.39 m/s cố định; biến thiên P ∈ [90, 250] W, α ∈ [0.27, 0.7], k_ratio ∈ [0.012, 0.2]. Đầu vào mạng: (x, y, z, t, P, α, k_ratio), tổng 7 chiều. Lý do giới hạn số chiều: PINN với nhiều tham số cần nhiều điểm huấn luyện và khó hội tụ.

**26.3 Dữ liệu FV.** Cần sửa `run_charles` để lưu trường nhiệt tại các mốc thời gian (ví dụ 10 mốc) và trên các lát cắt (bề mặt, và mặt cắt y = Y_TIP). Kích thước mỗi lượt lưới 10 µm: khoảng 137 nghìn ô × 10 mốc, đủ nhỏ để lưu. Thiết kế lấy mẫu: lưới Latin hypercube hoặc tổ hợp đầy đủ 3 × 3 × 3 tham số, tổng khoảng 27 lượt FV, mỗi lượt khoảng một phút trên GPU. Tách train/test theo tổ hợp tham số (không theo điểm không gian) để đo khả năng nội suy.

**26.4 Kiến trúc.** Mạng FNN, đầu vào chuẩn hóa về [−1, 1], đầu ra θ = (T − T_amb)/ΔT với output transform softplus để θ ≥ 0 (ràng buộc nonneg của mục 18). Điều kiện ban đầu lấy từ dữ liệu t = 0 (soft IC, không dùng hard IC vì cần biểu diễn chính xác các tham số bột).

**26.5 Hàm mất mát.** L = w_d·L_data + w_pde·L_PDE + w_bc·L_bc + w_E·L_E.
- L_data: MSE trên các điểm FV đã lưu.
- L_PDE: residual của phương trình enthalpy với k(T), c(T), tham số bột theo Charles eq. 14–15 và Moser (cùng định nghĩa như FV), tại các điểm lấy mẫu ngẫu nhiên và tập trung quanh chùm.
- L_bc: biên trên (laser bề mặt, đối lưu, bức xạ, bay hơi) tại điểm trên mặt; đáy T = T_amb; biên bên cách nhiệt.
- L_E: phạt sai lệch cân bằng năng lượng toàn miền, tính theo cùng công thức diag_star/FV (đảm bảo đo bằng công cụ đã kiểm chứng).
- Trọng số ban đầu: w_d = 1, các trọng số còn lại điều chỉnh theo kích thước residual; không dùng L-BFGS trước khi loss ổn định.

**26.6 Lộ trình huấn luyện theo tầng (mỗi tầng phải đạt trước khi sang tầng sau).**
- Tầng 1: dẫn nhiệt với bột và laser bề mặt, không bay hơi, không nhiệt ẩn. So với FV cùng cấu hình.
- Tầng 2: thêm bay hơi trên mặt (Cook eq. 11–12, làm trơn gần điểm sôi nếu cần).
- Tầng 3: thêm nhiệt ẩn nóng chảy (làm trơn f_l, theo mục 22).

**26.7 Tiêu chí chấp nhận (sửa theo đề xuất của thiết kế, cần người nghiên cứu xác nhận).**
- Mặt trên: sai số tương đối ≤ 2% so với FV, tại thời điểm cực đại.
- **Điểm B (đã duyệt ngày 2026-10-12): sai số tuyệt đối ≤ 2 K hoặc 2% của ΔT_B = T_B − T_amb, lấy giá trị lớn hơn.** Lý do: tiêu chí cũ tính trên nhiệt độ tuyệt đối (2% của 300 K ≈ 6 K) quá lớn so với mức tăng 12 K tại B. Bản gốc của đề xuất: cho sai số khoảng 6 K tại 300 K, quá khắt khe vì nhiệt tăng tại B chỉ 12 K. Đề xuất: sai số tuyệt đối ≤ 2 K trên mức tăng so với môi trường (ΔT_B), hoặc sai số ≤ 2% của ΔT_B. Điều này phải được xác nhận vì tiêu chí mục 25.2 đã ghi là "2% tại B".
- Cân bằng năng lượng của surrogate ≤ 5%.

**26.8 Rủi ro và giới hạn.**
- Ở điều kiện FV hiện tại, B không đạt nóng chảy trong toàn bộ dải tham số. Surrogate chủ yếu xác nhận điều này, ít khả năng phát hiện vùng có nóng chảy. Đây là hạn chế về giá trị, không phải lỗi thiết kế.
- Bay hơi cứng và tương phản bột/rắn làm PINN khó hội tụ, nên tầng 2 và 3 có nguy cơ không đạt.
- Thời gian huấn luyện PINN có thể lâu hơn FV trên GPU, nên lợi ích chỉ có khi cần đánh giá hàng nghìn tổ hợp tham số.

**26.9 Việc cần làm trước khi huấn luyện.**
1. Sửa `run_charles` để lưu trường tại các mốc và lát cắt.
2. Chạy sweep 27 lượt FV trên GPU.
3. Viết mã PINN tầng 1 và so với FV trên tập test.

## 27. Benchmark PINN thuần cho Ransenigo et al. 2022 (ngày 2026-10-12, quyết định của người nghiên cứu)

**27.1 Trường hợp.** 300 W, 1.75 m/s, bán kính chùm r_b = 50 µm (đường kính 100 µm, Bảng 2 và Hình 3 của bài), hấp thụ 0.4 (Bảng 2), nền bulk có lớp bột phẳng dày 60 µm (z > 180 µm, cùng hình học FV Z_POW). Tiết diện 1500 × 380 × 240 µm, quét từ x = −550 đến 650 µm.

**27.2 Vật lý phiên bản 1.** Dẫn nhiệt 3D với bột (Moser, k_ratio 0.05, porosity 0.405), laser bề mặt Gaussian, bay hơi theo Cook với T_cap = T_c, đối lưu và bức xạ (h = 25, ε = 0.2, như FV). Không có nhiệt ẩn, không có Marangoni, không có áp suất phản lực, không có VOF. Tính chất hằng số tại T_sol (ρ 4106, c_p 838, k 33.1).

**27.3 Bảng 5 (đã xác nhận ngày 2026-10-12).** T_sol = 1877 K, T_liq = 1933 K, T_b = 3533 K. Nhiệt bay hơi L_v = 900 J/g, đổi sang SI là **9.0 × 10⁵ J/kg (0.90 MJ/kg)**. Giá trị này thay cho L_v(0) = 9.83 MJ/kg của Cook chỉ trong trường hợp R2022. Chỉ chạy một giá trị này, không so sánh với giá trị khác. Lưu ý: công thức Cook dùng L_v trong hệ số mũ của áp suất hơi, nên đổi L_v0 cũng đổi hệ số mũ đó.

**27.4 Không đánh giá theo tiêu chí.** Các tiêu chí của mục 17, 25.2 và 26.7 không áp dụng cho trường hợp này. Kết quả chỉ được báo cáo dưới dạng chẩn đoán: cân bằng năng lượng, residual PDE, rộng và sâu vũng tại đường đẳng nhiệt T_sol, so với Bảng 3 của bài (rộng 84.4 ± 12.6 µm, sâu 68.0 ± 10.0 µm). Không đặt ngưỡng đạt hay không đạt.

**27.5 Không dùng FV làm mốc.** Lưới FV chưa hội tụ (xem README của `results/fv_exp4_dx5/`), nên mọi so sánh cho trường hợp này là với bài báo.

**27.6 Kiểm tra trước khi chạy.** Trước khi chạy trên GPU: kiểm tra toán tử PDE và biên trên các trường giải tích, kiểm tra hàm đo vũng trên trường tổng hợp, và chạy thử đầu cuối trên CPU với số vòng nhỏ. Các kiểm tra này là gỡ lỗi mã, không phải tiêu chí chấp nhận.

**27.7 Thực thi.** Huấn luyện chính chạy trên GPU của người nghiên cứu. Phiên làm việc của trợ lý không có GPU.

**27.8 Luật bay hơi (ngày 2026-10-12, đã duyệt).** Tổn thất nhiệt do bay hơi giữ theo Cook (q_evap, công thức của `physics.evaporation_flux`, với L_v = 9.0 × 10⁵ J/kg). Luật áp suất hơi tham chiếu Wu và cộng sự (2018, *J. Mater. Process. Technol.* 254, eq. 13): P_s(T) = P₀·exp[L_v(T − T_b)/(R_v·T·T_b)]. Hệ số mũ của Cook có cùng dạng, với L_v·M/R = L_v/R_v và R_v = R/M = 178 J/(kg·K) (M = 0.0467 kg/mol). Đây là dạng đúng khi L_v tính bằng J/kg; eq. 13 của Wu viết R không rõ đơn vị. Áp suất phản lực P_R ≅ 0.54·P_s (Wu eq. 12) không đưa vào PINN nhiệt vì nó chỉ tác động lên hình dạng bề mặt và dòng chảy. Bài Wu không cho công thức tổn thất nhiệt bay hơi, nên mục này không thay đổi công thức Cook.

**27.9 Kẹp bề mặt tại T_b (ngày 2026-10-12, quyết định của người nghiên cứu).** Trong vết laser (|x − x_l| < r_b, |y − y_tip| < r_b) nhiệt độ mặt trên được kẹp ở T_b = 3533 K (Bảng 5), dưới dạng điều kiện Dirichlet mềm trong loss. Ngoài vết laser giữ điều kiện thông lượng như cũ. Đây là xấp xỉ có chủ đích thay cho cân bằng thông lượng tại vết laser, vì mô hình bay hơi hiện tại quá yếu để giới hạn nhiệt độ (mục 27.8). Hệ quả: năng lượng không đóng ở biên trên trong vết laser, nên sai số năng lượng và thông lượng ngầm (suy ra từ θ) phải được báo cáo. Kết quả không so với tiêu chí đạt/không đạt (mục 27.4).

**27.10 Chuẩn hóa residual biên trên ngoài vết laser theo thang nhiệt độ (ngày 2026-10-12, quyết định của người nghiên cứu).** Trong chế độ kẹp (27.9), residual biên trên ngoài vết laser được chia cho θ_b = (T_b − T_amb)/T_scale, tức thang nhiệt độ, thay vì đỉnh thông lượng laser. Lý do: chia cho đỉnh thông lượng làm residual vùng xa rất nhỏ nên mạng không bị ràng buộc ở đó (xem kết quả lượt có điểm xa). Residual trong vết laser giữ nguyên dạng kẹp θ − θ_b. Không đổi tiêu chí (mục 27.4).

**27.11 Ràng buộc biên cứng cho vết laser (ngày 2026-10-12, quyết định của người nghiên cứu).** Thay penalty kẹp (27.9) bằng ràng buộc cứng theo hàm khoảng cách, theo Zhu và cộng sự (2021), *Comput. Mech.* 67, eq. 16–19:

- θ(x, t) = θ_bc(t)·(1 − H) + θ_raw·H, trong đó θ_raw là đầu ra mạng đã có ràng buộc IC và đáy.
- d = √( max(|x − x_l| − r_b, 0)² + max(|y − y_tip| − r_b, 0)² + (L_z − z)² ) là khoảng cách tới vùng kẹp trên mặt trên, với x_l(t) = x_start + v·t.
- H = ½(1 − cos(πd/δ)) khi d < δ, H = 1 khi d ≥ δ. Hệ số ½ bắt buộc để H liên tục bằng 1 tại d = δ (công thức không có ½ cho giá trị 2 tại d = δ). Cần đối chiếu với eq. 16 trong bản PDF.
- θ_bc(t) = θ_b·min(t/(ramp_length/v), 1), có đoạn khởi động như laser, để không mâu thuẫn với IC θ = 0 tại t = 0.
- δ = 10 µm, kiểm tra độ nhạy trong khoảng 5–20 µm.

Hệ quả: trong vết laser trên mặt, θ = θ_b tuyệt đối (sau khởi động). Hạng biên trên chỉ tính ngoài vết laser. Sai lệch có chủ đích: lớp chuyển tiếp dày δ dưới và quanh vết laser. Không đổi tiêu chí (mục 27.4). Ràng buộc này không giải quyết độ sâu vũng, cần kiểm tra riêng.

**27.12 Nguồn thể tích hình nón thay cho nguồn bề mặt (ngày 2026-10-12, thử nghiệm có chủ đích).** Theo nguồn hai nón của Piandoro và cộng sự (2026), *J. Manuf. Sci. Eng.* 148, mục 2.3, với các điểm sau:

- Hình học: z_t = 240 µm (mặt trên), z_i = z_t − H_u, z_b = z_i − H_l. Bán kính r_t (mặt), r_i (giữa), r_b (đáy). Mặc định: r_t = 50 µm (bán kính chùm, Bảng 2), r_i = 25 µm, r_b = 10 µm, H_u = H_l = 40 µm. Các giá trị này được chọn trước khi chạy, **không tinh chỉnh theo kết quả**, và không hiệu chỉnh theo Bảng 3. Kết quả so với Bảng 3 là đầu ra của thử nghiệm, không phải hiệu chuẩn.
- Bán kính r(z): nón dưới r = r_i − (r_i − r_b)(z_i − z)/(z_i − z_b); nón trên r = r_t − (r_t − r_i)(z_t − z)/(z_t − z_i).
- Phân bố năng lượng: q(x, y, z, t) = (ηP·ramp/H) · 3/(πr²) · exp(−3ρ²/r²), với ρ² = (x − x_l)² + (y − y_tip)², H = H_u + H_l, trong vùng z_b ≤ z ≤ z_t. Mỗi lát sâu nhận cùng năng lượng, nên ∫q dV = ηP. Hệ số 3/(πr²) được chọn để tích phân theo mặt bằng 1 (thay cho hệ số e³ của bài báo, vốn bị mất trong bản trích).
- Không có thông lượng laser trên mặt trên; biên trên chỉ có đối lưu, bức xạ và bay hơi (như trước).
- Vùng chuyển tiếp của ràng buộc cứng (27.11) không dùng với nguồn này. Chỉ chạy surface = flux.

**Kiểm tra trước khi chạy:** bán kính tại biên, ∫q dV = ηP, PDE có hạng nguồn, biên trên không có thông lượng laser khi dùng nguồn thể tích.

**27.12 (bổ sung, ngày 2026-10-12).** Với nguồn hình nón, đỉnh Q* tại đáy nón (r_b = 10 µm) khoảng 270, lớn hơn θ hàng trăm lần, nên hạng PDE không cân bằng với các hạng biên. Chuẩn hóa residual PDE theo đỉnh nguồn Q*_pk = 3ηP/(Hπr_b²)·L_c²/(k_c T_scale) trong chế độ hình nón. Đây là thay đổi trọng số đã ghi trước, không đổi hình học hay tiêu chí.

**27.12 Kết quả thử (ngày 2026-10-12, không phải kết quả chính).** Nguồn hình nón (r_t 50, r_i 25, r_b 10 µm; H = 80 µm) với 6000 vòng CPU: nhiệt độ mặt tâm khoảng 7870 K, T_max lưới 8140 K, năng lượng tích lũy 1.44 J so với đầu vào 0.082 J (rel_error −18), vũng chiếm toàn bộ miền. Với 300–1500 vòng: nhiệt độ khoảng 650 K, rel_error −1.65, không có vũng. Kết luận: trong công thức hiện tại (mất nhiệt đối lưu và bức xạ, bay hơi Cook với L_v = 9·10⁵ J/kg), biên trên không cân bằng được nguồn thể tích. Mạng hoặc chưa hội tụ (ngắn) hoặc tích lũy năng lượng không giới hạn (dài). Cần một cơ chế cân bằng năng lượng toàn cục (hạng L_E, như mục 26.5) hoặc giới hạn bề mặt trước khi so với Bảng 3. Không đổi tiêu chí.

**27.13 Ràng buộc năng lượng toàn cục L_E (ngày 2026-10-12, quyết định của người nghiên cứu).** Thêm hạng L_E = ((E_in − F_top − F_bot − E_stored)/E_in)² vào loss, với E_in = αP(t_end − ramp/2v), F_top và F_bot ước lượng bằng trung bình Monte Carlo trên điểm mặt trên và đáy đều theo không gian và thời gian, E_stored = ∫ρc·g·T_scale·θ dV tại t_end ước lượng bằng Monte Carlo. Bỏ qua thông lượng mặt bên (đã kiểm tra nhỏ, dưới 10⁻⁴ J trong các chẩn đoán). Trọng số w_E được đặt trước (mặc định 0, tắt). Đây là ràng buộc vật lý, không phải hiệu chỉnh để khớp Bảng 3, và không đổi tiêu chí (mục 27.4).

**27.13 (bổ sung, ngày 2026-10-12).** Với L_E, nghiệm vẫn gần đồng đều vì hạng PDE quá nhẹ so với các hạng khác. Thêm trọng số w_pde (tham số --w-pde, mặc định 1.0) nhân với hạng PDE. Thử w_pde = 10 với nguồn hình nón và w_E = 1. Không đổi hình học, vật lý hay tiêu chí.

**27.13 (kết quả thử w_pde = 10, ngày 2026-10-12).** Nghiệm không đổi (nhiệt độ cực đại ~883 K, rel_error −0.004, hạng PDE cuối 6.96·10⁻⁴ với cả w_pde = 1 và 10). Gradient cho thấy nguyên nhân: gradient của L_E khoảng 0.32, còn gradient PDE chỉ khoảng 2.5·10⁻⁶ (đã có chuẩn hóa theo đỉnh Q*). Hạng PDE có trọng số thực tế quá nhỏ, và các điểm lấy mẫu gần như không nằm trong lõi nguồn. Kết luận: tăng trọng số không đủ. Cần lấy mẫu dày trong lõi nguồn hoặc đổi cách trung bình residual theo Q. Không đổi tiêu chí.

**27.14 Lấy mẫu lõi nguồn và trọng số PDE theo Q (ngày 2026-10-12, quyết định của người nghiên cứu).** Với nguồn hình nón, thêm n_core điểm miền lấy đều trong nón: z đều trong [z_b, z_t], (x, y) đều trong ±r(z) quanh tâm laser tại thời điểm đó. Hạng PDE (chuẩn hóa theo Q*_pk, 27.13 bổ sung) được nhân với trọng số w = 1 + Q*/Q*_pk, trong đó Q* là nguồn tại điểm đó (không tính đạo hàm qua w). Lý do: các điểm gần lõi nguồn hiện chiếm tỷ lệ quá nhỏ, nên residual PDE tại lõi bị che khuất (xem 27.13 kết quả thử). Không đổi hình học, vật lý hay tiêu chí (27.4).

**27.15 Biên bên đối lưu–bức xạ và lấy mẫu Sobol (ngày 2026-10-12, quyết định của người nghiên cứu, theo Faegh và cộng sự 2026).** (a) Bốn mặt bên: thay cách nhiệt bằng đối lưu và bức xạ, −k∂T/∂n = h(T − T_amb) + εσ(T⁴ − T_amb⁴) với cùng h và ε của mặt trên (không có bay hơi trên mặt bên). Lý do: điểm nóng sát biên y ≈ 0 trong các lượt trước có thể do biên cách nhiệt. (b) Lấy mẫu bằng dãy Sobol có xáo trộn cho điểm miền gần vết laser; mật độ theo z dày hơn gần mặt trên (z = L_z(1 − U²)). Tùy chọn --sobol, mặc định tắt. Không đổi hình học, tiêu chí (27.4) hay các thay đổi khác. Chạy thử trên CPU 3000 vòng so với lượt cone_core.

**27.15 (kết quả thử, ngày 2026-10-12).** Nguồn hình nón + L_E + lõi nguồn + biên bên Robin + Sobol, 3000 vòng CPU: nhiệt độ cực đại ~890 K, rel_error −0.001, không có vũng, hạng PDE ~9·10⁻³. So với lượt không có biên Robin/Sobol (cone_core): gần như giống nhau. Kết luận: biên bên và lấy mẫu Sobol không giải quyết việc mạng không biểu diễn được nguồn tập trung. Không đổi tiêu chí.

**27.12 Rút lại (ngày 2026-10-12, quyết định của người nghiên cứu).** Nguồn thể tích hình nón bị gỡ khỏi mã. Đường đi chính là nguồn bề mặt Gaussian như trước. Các kết quả thử của nguồn hình nón (mục 27.12, 27.13, 27.14 và các lượt CPU có tên `cone` trong `results/`) được giữ làm lịch sử, không dùng làm kết quả. Các tùy chọn chung (L_E, biên bên Robin, Sobol, trọng số PDE) vẫn có trong mã, mặc định tắt.

**27.16 Quay về cấu hình chính (ngày 2026-10-12, quyết định của người nghiên cứu).** Cấu hình chính của PINN R2022 là: nguồn Gaussian bề mặt (source = surface), ràng buộc cứng trong vết laser (surface = hard, δ = 10 µm, mục 27.11), biên bên cách nhiệt (side_bc = adiabatic), không có L_E (w_energy = 0), giảm learning rate ở cuối (mục 27.9–27.10 và lượt lrdecay). Các nhánh thử (nguồn hình nón 27.12, L_E 27.13, lõi nguồn và trọng số PDE 27.14, biên Robin và Sobol 27.15) vẫn được giữ trong mã làm tùy chọn, không dùng làm kết quả chính, và không nằm trong báo cáo đạt/không đạt. Nhánh hình nón không cho nghiệm vật lý ổn định, nên không đưa vào kết quả chính. Không đổi tiêu chí (27.4).

**27.17 Bốn biên bên đối lưu–bức xạ trong cấu hình chính (ngày 2026-10-12, theo Faegh và cộng sự 2026).** Thử ràng buộc cứng (27.11) với biên bên cách nhiệt và biên bên Robin, 3000 vòng CPU. Kết quả: rộng 224 µm cả hai; sâu 30 µm (cách nhiệt) và 32 µm (Robin); ở 250 µm, 774 K và 704 K; điểm sát biên y ≈ 10 µm: 3594 K và 3472 K. Biên Robin chỉ thay đổi nhỏ. Các lượt CPU 3000 vòng chưa hội tụ nên không so với lượt GPU 20000 vòng. Không đổi tiêu chí (27.4).

**27.18 Kiểm tra độ lặp lại theo seed (ngày 2026-10-12, quyết định của người nghiên cứu).** Cấu hình chính (ràng buộc cứng, 27.16–27.17) chạy trên GPU 20 000 vòng với 3 seed (0, 1, 2), cho cả hai lựa chọn biên bên (cách nhiệt, Robin). Tổng 6 lượt. Chỉ so sánh trung bình và độ lệch chuẩn của rộng, sâu, nhiệt độ vùng xa và rel_error; chênh lệch giữa hai lựa chọn chỉ được coi là có ý nghĩa khi vượt độ lệch chuẩn theo seed. Không đổi tiêu chí (27.4).

**27.19 Quay về điều kiện thông lượng có bay hơi, bỏ kẹp T_b (ngày 2026-10-12, quyết định của người nghiên cứu).** Bỏ ràng buộc kẹp (27.9, 27.11) trong vết laser. Dùng điều kiện thông lượng trên mặt trên với bay hơi Cook, đối lưu và bức xạ (surface = flux, source = surface, side_bc = adiabatic). Lý do: Wu và cộng sự (2018) cho thấy bay hơi làm vũng hẹp và sâu hơn; kẹp T_b loại bỏ cơ chế này ở vết laser. Chạy trên CPU 3000 vòng với hai giá trị L_v: 9.0·10⁵ J/kg (Bảng 5) và 9.83·10⁶ J/kg (Cook). Tiêu chí kiểm tra: nếu với Cook nhiệt độ mặt được giới hạn gần T_b và vũng hẹp lại, lỗi chính là giá trị L_v. Không đổi tiêu chí chấp nhận (27.4).

**27.19 Kết quả thử (ngày 2026-10-12).** Thông lượng có bay hơi, không kẹp, 3000 vòng CPU, với L_v = 9.0·10⁵ và 9.83·10⁶ J/kg. Cả hai lượt cho nghiệm nguội (~320 K, không vũng, sai số năng lượng ~0.92). Kết quả không phụ thuộc L_v, nên không kết luận được về giá trị L_v. Nguyên nhân khả dĩ: trình tối ưu kẹt ở nghiệm nguội như đã gặp trước đó. Cần phương pháp khởi động (ví dụ tiếp tục hóa có tắt cả bức xạ lẫn bay hơi ở giai đoạn đầu) trước khi so sánh L_v.

**27.19b Tiếp tục hóa cho bay hơi (ngày 2026-10-12, quyết định của người nghiên cứu).** Để thoát nghiệm nguội, giai đoạn đầu (1500 vòng) chỉ có thông lượng laser, không đối lưu, bức xạ hay bay hơi. Sau đó tăng tuyến tính (1500 vòng) hệ số đối lưu, bức xạ và bay hơi (φ nhân với frac). Cuối cùng 1000 vòng với vật lý đầy đủ. Chạy cho cả hai L_v, so sánh nhiệt độ mặt, vũng và sai số năng lượng. Không đổi tiêu chí (27.4).

**27.19c Cân bằng biên thông lượng với PDE (ngày 2026-10-12, quyết định của người nghiên cứu).** Ở tiếp tục hóa, loss biên thông lượng vẫn ở mức nghiệm nguội (~0.0066) ngay cả khi không có mất nhiệt, nên hạng biên bị lấn át bởi PDE. Thử tăng trọng số hạng biên trên (w_top) lên 1 và 100 với L_v = 9·10⁵, không tiếp tục hóa, 3000 vòng CPU. Tiêu chí kỹ thuật: loss biên thông lượng phải xuống dưới mức nghiệm nguội. Không đổi tiêu chí chấp nhận (27.4).

**27.20 Xác nhận tham số, bỏ L_E, và kiểm tra cấu hình theo bài báo (ngày 2026-10-13, quyết định của người nghiên cứu).**

(a) Tham số R2022 đã xác nhận: L_v = 900 J/g = 9.0·10⁵ J/kg (Bảng 5 của Ransenigo và cộng sự 2022); L_f = 338.8 J/g (Bảng 5); T_sol = 1877 K, T_liq = 1933 K, T_b = 3533 K (Bảng 5). Nhiệt độ môi trường T_amb = 293 K theo quyết định của người nghiên cứu (bài báo dùng 298 K ở đáy, Hình 5). Giá trị L_v của Cook (9.83·10⁶ J/kg) không dùng trong R2022; mã R2022 không tham chiếu giá trị này (`--lv0` chỉ còn mặc định 9.0·10⁵). Mã FV và `config.py` giữ nguyên, không dùng cho kết luận.

(b) Bỏ L_E: xóa hàm `energy_penalty`, `sample_energy`, các tham số `--w-energy` và `--n-energy`, và nhánh trong vòng huấn luyện. Mục 27.13 (L_E) được thay thế bởi quyết định này; không còn trong mã. Các lượt có L_E trước đây giữ làm lịch sử.

(c) Bỏ áp suất phản lực (recoil): không có trong mô hình R2022 và không có trong tài liệu này. Bài Ransenigo và cộng sự 2022 nêu áp suất phản lực bay hơi có trích dẫn Wu và cộng sự 2018, nhưng mô hình không đưa vào. Cũng không có nguồn thể tích hình nón (đã gỡ ở 27.12).

(d) Kiểm tra cấu hình theo các bài báo đã tải lên. Nhãn: **[có bằng chứng]** = bài báo đã tải lên có nêu; **[suy diễn]** = do thiết kế của chúng ta, bài báo không nêu.

| Mục cấu hình | Nhãn | Bằng chứng / ghi chú |
|---|---|---|
| Ràng buộc cứng IC và đáy (θ = z·softplus·(1−e^{−t})) | [suy diễn] | Zhu và cộng sự 2021 dùng ràng buộc cứng Dirichlet (hàm Heaviside) nói chung; dạng cụ thể của chúng ta là thiết kế riêng. |
| Ràng buộc cứng bề mặt trong vết laser θ = θ_b (27.11) | [suy diễn] | Không có bài nào đã tải lên dùng kẹp nhiệt độ bề mặt trong vết laser. Zhu và cộng sự dùng ràng buộc cứng cho biên Dirichlet nhưng không cho bề mặt laser. Kết quả đạt chính xác (sai số dư ~10⁻¹² K) là do cấu trúc, không phải bằng chứng vật lý. |
| Đáy Dirichlet T = T_amb | [có bằng chứng] | Ransenigo và cộng sự (Hình 5, 298 K); Faegh và cộng sự 2026 (T = T_amb); Piandoro và cộng sự 2026 (298 K); Sharma và cộng sự 2026 (298 K). Giá trị 293 K là quyết định của người nghiên cứu. |
| Biên bên cách nhiệt (mặc định) | [có bằng chứng] một phần | Piandoro và cộng sự 2026 dùng cách nhiệt cho biên ngoài. Faegh và cộng sự 2026 dùng Robin (đối lưu, bức xạ) cho cả bốn mặt bên (chúng ta có tùy chọn Robin 27.15). |
| Biên trên: thông lượng Gaussian + đối lưu + bức xạ | [có bằng chứng] | Faegh và cộng sự 2026 (Hình 2); Piandoro và cộng sự 2026 (h = 20, ε = 0.9). Hệ số h = 25, ε = 0.2 của chúng ta lấy từ FV/Charles, không từ bài đã tải lên. |
| Bay hơi Cook trên mặt trên | [suy diễn] | Cook không có trong các bài đã tải lên. Faegh và cộng sự 2026 bỏ bay hơi ("conduction-mode only"); Sharma và cộng sự 2026 không xét bay hơi. Ransenigo và cộng sự nêu bay hơi/áp suất phản lực nhưng không công thức Cook. |
| Trường nhiệt dẫn có bột (Moser, κ, g) | [suy diễn] | Phương trình dẫn nhiệt tổng quát có trong các bài; hệ số bột Moser không có trong bài đã tải lên. Dùng như FV. |
| Nhiệt ẩn nóng chảy 338.8 J/g | [có bằng chứng] (ở FV) | Bảng 5 của Ransenigo và cộng sự 2022. PINN R2022 không có nhiệt ẩn (không có thuật ngữ chuyển pha). Piandoro và cộng sự 2026 và Zhou và cộng sự 2026 có đưa nhiệt ẩn vào; Sharma và cộng sự 2026 bỏ qua. Đây là hạn chế đã biết. |
| Nhiệt ẩn bay hơi 900 J/g | [có bằng chứng] | Bảng 5 của Ransenigo và cộng sự 2022. |
| Adam, learning rate, lịch giảm | [có bằng chứng] một phần | Sharma và cộng sự 2026: Adam lr 2·10⁻⁴ rồi L-BFGS. Faegh và cộng sự 2026: bảng cấu hình ADAM lr 10⁻³, 5 lớp × 32 nơron, tanh. Piandoro và cộng sự 2026: L-BFGS. Số 20 000 vòng và lịch giảm tuyến tính **[suy diễn]**, không có trong bài đã tải lên. Cấu hình chúng ta: 4 × 64, tanh, lr giảm tuyến tính. |
| Lấy mẫu Sobol và tập trung quanh laser | [có bằng chứng] (tùy chọn 27.15) | Faegh và cộng sự 2026 dùng Sobol và lấy mẫu quanh laser. Cấu hình chính hiện không dùng Sobol (27.16). |
| Cân bằng năng lượng toàn cục (L_E) | [suy diễn] và đã bỏ | Không có trong bài đã tải lên dưới dạng hạng loss. Piandoro và cộng sự 2026 dùng nhiệt dung biểu kiến; Zhu và cộng sự 2021 có bảo toàn năng lượng trong loss. Đã bỏ theo (b). Chẩn đoán cân bằng năng lượng (energy_budget) vẫn là kiểm tra đầu ra, không phải hạng loss. |
| Áp suất phản lực | [có bằng chứng] nhưng không dùng | Wu và cộng sự 2018 (P_R ≅ 0.54 P_s), được Ransenigo và cộng sự 2022 trích dẫn. Không đưa vào mô hình R2022 (c). |

(e) FV (fv_stage0): theo yêu cầu của người nghiên cứu, không dùng trong kết luận. Mã và kết quả FV chưa xóa; xóa cần xác nhận riêng.

(f) Kết quả đạt/không đạt của cấu hình chính (mục 27.16–27.18, các số liệu trước khi đổi T_amb sang 293 K và trước khi bỏ L_E): vũng quá rộng (~157–169 µm so với 84.4 ± 12.6 µm) và quá nông (~29 µm so với 68.0 ± 10.0 µm). Các lượt thông lượng có bay hơi (27.19, 27.19b) không học được điều kiện laser: loss biên dừng ở ~0.0066–0.0074, nghiệm nguội. Không có dữ liệu nhiệt độ thực nghiệm để kiểm chứng trường nhiệt độ. Các số này chưa tái chạy với T_amb = 293 K; cần chạy lại để có số mới.

(g) Không đổi tiêu chí chấp nhận (27.4).

**27.21 Kiểm tra ba bài đã tải lên làm quy chuẩn PINN, bỏ bay hơi (ngày 2026-10-13, quyết định của người nghiên cứu).**

Ba bài: Yuan và cộng sự 2025 (IJAMT, PINN 3D L-DED, đa đường, dữ liệu hồng ngoại; gọi là bài A), Faegh và cộng sự 2026 (Additive Manufacturing, PINN cho đường quét bất kỳ, gọi là bài B), Zhu, Liu, Yan 2021 (Comput. Mech., PINN với ràng buộc cứng Dirichlet bằng hàm Heaviside, gọi là bài C).

(a) Ràng buộc cứng kẹp T:
- Bài C đề xuất ràng buộc cứng Dirichlet bằng hàm Heaviside: u = u_bc(1 − H) + uH (Eq. 17–19), với H = 1 − cos(dπ/ε) khi d < ε, và H = 1 khi d ≥ ε (Eq. 16). Bài chỉ kiểm chứng trên bài toán 1D đông đặc có nghiệm giải tích (Fig. 4), cho thấy ràng buộc cứng học nhanh hơn và chính xác hơn ràng buộc mềm. Công thức in trong Eq. 16 không liên tục tại d = ε (H → 2⁻ rồi nhảy về 1); dạng ½(1 − cos) của chúng ta là dạng liên tục từ 0 đến 1, nên phần ½ là cần thiết và không phải sai khác tùy ý. Nhãn: **[có bằng chứng]** cho ràng buộc cứng Dirichlet trên biên của miền.
- Bài B dùng ràng buộc cứng cho điều kiện ban đầu (Eq. 18–19: T = T_PINN(t > 0) + T_0(t = 0)). Biên đáy T = T_amb (Eq. 7) và các biên bên là ràng buộc mềm (L_BC, Eq. 17). Nhãn: **[có bằng chứng]** cho ràng buộc cứng IC; biên đáy là mềm trong bài B.
- Bài A dùng ràng buộc mềm cho IC và biên (Eq. 4, trọng số w_ic = 10/12, w_bc = 1/12). Bài A không có ràng buộc cứng.
- Không bài nào trong ba bài kẹp nhiệt độ bề mặt trong vết laser (T = T_b trong vùng laser). **[suy diễn]**. Vì vậy ràng buộc cứng kẹp T trong vết laser (mục 27.11) không có bằng chứng từ ba bài.
- Kết luận: ràng buộc cứng IC và đáy có bằng chứng (bài B, C). Ràng buộc cứng kẹp T trong vết laser không có bằng chứng; **chờ quyết định của người nghiên cứu** (xem (d)).

(b) Bay hơi: bỏ.
- Bài A: không đưa bay hơi vào mô hình. Bài B: Table/đoạn cuối phần 2.1 nói "the latent heat of fusion and vaporization ... are not considered in the current study", và phần kết luận ghi bay hơi thuộc phạm vi tương lai (keyhole). Bài C: giả thiết "the loss of metal material due to vaporization ... negligible" (mục 2.1), và phần thảo luận: "the evaporation phenomenon ... not crucial for the applications considered". Không bài nào có công thức bay hơi Cook hay áp suất phản lực.
- Ghi chú: bài mục tiêu Ransenigo và cộng sự 2022 nêu nhiệt ẩn bay hơi 900 J/g và trích dẫn áp suất phản lực (Wu và cộng sự 2018). Đó là bằng chứng cho bài toán mục tiêu, không nằm trong ba bài này. Quyết định bỏ bay hơi nghĩa là bỏ cơ chế đó khỏi mô hình R2022; L_v = 9.0·10⁵ J/kg vẫn lưu trong `R2022Case` nhưng không còn được dùng.
- Thay đổi mã: `R2022Case.material()` đặt `phi_evap = 0.0`, nên hạng bay hơi Cook trong `top_outflux` bằng 0. Hệ số `phi_scale` vẫn nằm trong mã nhưng không còn tác dụng (tiếp tục hóa 27.19b trở thành không có hiệu ứng bay hơi).

(c) Các mục khác (đối chiếu với ba bài):
- Biên trên: Gaussian + đối lưu + bức xạ. Bài B (Eq. 2–3) và bài A (Eq. 2b, nguồn thể tích Gaussian ellipsoid) đều có đối lưu và bức xạ. Bài C dùng thông lượng laser và biên không thông lượng cho các biên khác.
- Biên bên: bài B dùng đối lưu + bức xạ (Robin) cho bốn mặt; bài C dùng không thông lượng (adiabatic) cho các mặt không có laser (mục 4.2); bài A không nêu. Mặc định cách nhiệt của chúng ta có bằng chứng từ bài C; Robin có bằng chứng từ bài B.
- Đáy: bài B Dirichlet T = T_amb (Eq. 7, Table 1: 298 K); bài A Table 2: T_0 = 293.15 K. Bằng chứng cho 293 K có trong bài A.
- Lấy mẫu: bài B dùng Sobol với mật độ theo z dày hơn gần mặt trên (Eq. 9–11), và lấy mẫu quanh laser theo phân bố Gaussian (Fig. 4a). Đây là bằng chứng cho mục 27.15.
- Adam và learning rate: bài B dùng ADAM, lr 10e-4 (Table 3; ghi như vậy trong bảng), 5 lớp × 32 nơron, tanh, 100 000 epoch (mục 3.3.1, trong đó bài nói 100k epoch là điểm hội tụ). Bài A dùng Adam, 6 lớp × 50 nơron, 30 000 vòng (Table 4). Bài C dùng Adam, khoảng 25 000 vòng (Fig. 4). Không bài nào dùng lịch giảm learning rate tuyến tính. Nhãn: **20 000 vòng và giảm lr tuyến tính là [suy diễn]**; số vòng nằm trong khoảng 25 000–100 000 của các bài.
- Trường bột Moser, κ và g: không có trong ba bài. Bài B dùng khối đặc Inconel không có bột. **[suy diễn]**.
- Nhiệt ẩn nóng chảy: bài C có (Eq. 3, L = 290 kJ/kg cho nhôm/IN625); bài B nêu bỏ qua trong nghiên cứu hiện tại; bài A không có. Mô hình R2022 chưa có nhiệt ẩn: hạn chế đã biết.
- Kiểm chứng định lượng vũng nóng chảy: bài B so sánh định tính với thực nghiệm (Fig. 14, "consistent qualitative trends"); bài C có độ dài vũng so với thực nghiệm (782 ± 21 µm), còn độ rộng và độ sâu không có số đo thực nghiệm (Table 4, "–"). Bài A so với FEM, không có dữ liệu nhiệt độ thực nghiệm đo được bên trong. Không bài nào kiểm chứng định lượng cả độ rộng và độ sâu vũng cùng lúc.

(d) Quyết định cần người nghiên cứu xác nhận (không thay đổi trong mục này):
- Kẹp T trong vết laser (27.11, cấu hình chính hiện tại): không có bằng chứng từ ba bài. Nếu bỏ, cấu hình chính chuyển về thông lượng Gaussian + không bay hơi, đã biết là không học được (nghiệm nguội, 27.19 và 27.19b, trước khi bỏ bay hơi).
- Bay hơi bỏ theo (b) làm L_v không còn được dùng trong R2022; bài mục tiêu Ransenigo có L_v (Bảng 5) và trích dẫn Wu 2018. Cần xác nhận việc bỏ bay hơi khỏi bài toán mục tiêu có phù hợp không.

**27.22 Cấu hình chính mới: bỏ kẹp T trong vết laser, chạy lại (ngày 2026-10-13, quyết định của người nghiên cứu).**

(a) Cấu hình chính: thông lượng Gaussian trên mặt trên (surface = flux), không kẹp T (không dùng surface = hard, mục 27.11), không bay hơi (phi_evap = 0, mục 27.21), biên bên cách nhiệt (side_bc = adiabatic), T_amb = 293 K, không L_E, learning rate giảm tuyến tính từ 10⁻³ xuống 10⁻⁴ trong 5000 vòng cuối, 20 000 vòng, seed 0. Mã không đổi về mặc định (surface và side_bc đã mặc định là flux và adiabatic); chỉ đổi lệnh chạy.

(b) Lệnh: `python -m lpbfpinn.r2022.train --device cpu --surface flux --side-bc adiabatic --seed 0 --iters 20000 --lr 1e-3 --lr-final 1e-4 --decay-iters 5000 --out results/pinn_r2022_flux_T293`.

(c) Lưu ý về khác biệt với lượt chính cũ (pinn_r2022_hard_lrdecay): lượt cũ dùng kẹp T trong vết laser và T_amb = 298 K, có bay hơi. Giá trị lr cuối và số vòng giảm của lượt cũ không được ghi trong repo, nên các giá trị 10⁻³ → 10⁻⁴ và 5000 vòng là lựa chọn mới, không phải lặp lại từ lượt cũ.

(d) Chạy trên CPU (4 nhân, không có GPU trong môi trường này), khoảng 0.14 s/vòng. Kết quả và kiểm tra: diag.json, profile, và so sánh với mục tiêu Bảng 3 (vũng rộng 84.4 ± 12.6 µm, sâu 68.0 ± 10.0 µm). Tiêu chí chấp nhận không đổi (mục 27.4).

**27.22 (cập nhật, ngày 2026-10-13).** Lượt CPU 20 000 vòng đã dừng theo yêu cầu của người nghiên cứu, chưa hoàn thành; log `results/pinn_r2022_flux_T293.log` là lượt dừng sớm, không dùng làm kết quả. Lượt chính chuyển sang GPU của người nghiên cứu với cùng cấu hình: `python -m lpbfpinn.r2022.train --device cuda --surface flux --side-bc adiabatic --seed 0 --iters 20000 --lr 1e-3 --lr-final 1e-4 --decay-iters 5000 --out results/pinn_r2022_flux_T293`. Sau khi chạy xong, chạy diagnose và profile trên `results/pinn_r2022_flux_T293/model.pt`, rồi push diag.json, profile.json, history.json và log.

**27.23 Giới hạn đã biết: bay hơi và áp suất phản lực (ngày 2026-10-13, quyết định của người nghiên cứu).**

(a) Mô hình R2022 không có bay hơi (phi_evap = 0, mục 27.21) và không có áp suất phản lực trên mặt tự do. Kết quả vũng nóng chảy sẽ lệch khỏi Bảng 3 của Ransenigo và cộng sự 2022, và lệch này được ghi là giới hạn của mô hình, không phải là kết quả thất bại của tối ưu hóa.

(b) Cơ sở: Wu và cộng sự 2018 (Bảng 2 và Hình 8) cho thấy không có bay hơi, vũng 146 × 68 µm, nhiệt cực đại 5439 K; có bay hơi kèm phản lực, vũng 108 × 82 µm, nhiệt cực đại 2676 K. Cơ chế làm vũng hẹp và sâu là phản lực trên mặt tự do, cần bề mặt biến dạng (level set hoặc VOF), không nằm trong phạm vi PINN dẫn nhiệt thuần. Bay hơi Cook với L_v = 9·10⁵ J/kg chỉ khoảng 0.1% thông lượng đỉnh, nên không thay thế được cơ chế này.

(c) Không thêm hệ số hiệu chỉnh bay hơi để khớp kích thước vũng. Đối chứng Cook với hệ số lớn, nếu có, chỉ để kiểm tra độ nhạy và phải ghi là đối chứng, không phải cấu hình chính.

(d) Báo cáo: kích thước vũng và nhiệt cực đại của R2022 được so với Bảng 3 nhưng không kết luận là khớp hay không khớp mà không nêu giới hạn này.

**27.24 Thử hai hướng cho vấn đề PDE kéo nghiệm về trạng thái nguội (ngày 2026-10-13, quyết định của người nghiên cứu).**

Quan sát nền: lượt CPU 4000 vòng (2000 vòng biên trên trước, rồi cả hai hạng) cho nhiệt cực đại 455 K và không có vũng. Khi PDE bật ở vòng 2000, hạng PDE là 44.5 và sau đó giảm về 10⁻⁴, trong khi hạng biên trên quay từ 5·10⁻⁵ lên 0.0073. Giả thuyết: hạng biên trên trung bình trên tất cả điểm mặt trên, nên các điểm trong vết laser (một phần nhỏ) bị pha loãng, và mạng được thưởng khi giữ θ nhỏ để thỏa PDE.

(a) Hướng 1, `--spot-balance`: hạng biên trên = ½·trung bình r² trên điểm trong vết laser + ½·trung bình r² trên điểm ngoài vết. Không đổi vật lý, không dùng kẹp T.

(b) Hướng 2, `--pde-ramp-iters N`: trọng số PDE tăng tuyến tính từ 0 lên w_pde trong N vòng sau `--pre-iters` (mặc định 0). Không đổi vật lý.

(c) Lượt GPU (cấu hình chính, seed 0, 20 000 vòng, lr 10⁻³ → 10⁻⁴ trong 5000 vòng cuối, không kẹp T, không bay hơi, T_amb 293 K, biên bên cách nhiệt). Lượt đối chứng là `results/pinn_r2022_flux_T293` (mục 27.22, chưa chạy xong):
- A (hướng 1): `python -m lpbfpinn.r2022.train --device cuda --surface flux --side-bc adiabatic --seed 0 --iters 20000 --lr 1e-3 --lr-final 1e-4 --decay-iters 5000 --spot-balance --out results/pinn_r2022_flux_spotbal`
- B (hướng 2): `python -m lpbfpinn.r2022.train --device cuda --surface flux --side-bc adiabatic --seed 0 --iters 20000 --lr 1e-3 --lr-final 1e-4 --decay-iters 5000 --pde-ramp-iters 5000 --out results/pinn_r2022_flux_pderamp`

(d) Thử ngắn trên CPU (600 vòng, lấy mẫu nhỏ) đã chạy để kiểm tra mã: cả hai tùy chọn chạy được, hạng biên trên trong vết laser hiện lớn hơn khi cân bằng (0.05 so với 0.0066), nghĩa là sai số trong vết laser không còn bị che. Đây không phải kết quả.

(e) Tiêu chí đánh giá (mục 27.4 không đổi): một hướng được coi là có ích nếu, sau diagnose trên lượt GPU, (i) nhiệt cực đại trên lưới vượt T_sol = 1877 K, (ii) xuất hiện vũng nóng chảy (rộng và sâu lớn hơn 0), (iii) hạng biên trên trong vết laser (rms/peak) và hạng PDE đều nhỏ. Kết quả so với Bảng 3 được báo cáo đầy đủ, không chọn lượt theo kích thước mục tiêu.

**27.24 (kết quả CPU ngắn, ngày 2026-10-13).** Ba lượt CPU 2000 vòng, seed 0, miền thu nhỏ (n_domain 1000, n_top 400), không kẹp T, không bay hơi, T_amb 293 K:
- Đối chứng (không tùy chọn): nhiệt cực đại 330 K, bề mặt 326 K, không vũng, sai số năng lượng 0.885, hạng PDE cuối 7·10⁻⁶, hạng biên trên cuối 0.0089.
- `--spot-balance`: nhiệt cực đại 332 K, bề mặt 328 K, không vũng, sai số năng lượng 0.881, hạng PDE cuối 7·10⁻⁶, hạng biên trên cuối 0.102.
- `--pde-ramp-iters 1000`: nhiệt cực đại 297 K, không vũng, sai số năng lượng 0.988, hạng PDE cuối 0.
Kết luận: không hướng nào đạt tiêu chí 27.24(e) trong 2000 vòng CPU. Cả ba lượt vẫn nguội. Hạng biên trên của lượt cân bằng vết laser cao hơn nhiều (0.10 so với 0.009), cho thấy sai số trong vết laser không còn bị pha loãng, nhưng mạng chưa giảm được nó trong thời gian này. Chỉ số rms/peak trong vết laser của diagnose (0.443) giống nhau ở cả ba lượt, nên chưa dùng để so sánh được. Cần thử lâu hơn và kết hợp hai hướng trước khi kết luận.
