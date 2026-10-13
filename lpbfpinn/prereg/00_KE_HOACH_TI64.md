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

**27.24 (kết quả GPU, hướng 2 `--pde-ramp-iters 5000`, ngày 2026-10-13).** Lượt `results/pinn_r2022_flux_pderamp` (seed 0, 20 000 vòng, không kẹp T, không bay hơi, T_amb 293 K, biên bên cách nhiệt) chạy trên GPU của người nghiên cứu. Diagnose: nhiệt cực đại trên lưới 293.5 K, bề mặt trung tâm 293.5 K, không vũng (rộng và sâu 0 µm), sai số năng lượng 0.998, E_in 0.082 J so với E_stored 1·10⁻⁴ J. Profile trên mặt trên ở ba thời điểm: 293.4–293.5 K trên toàn bề rộng 250 µm. Rms/peak trong vết laser 0.443, giống các lượt nguội khác: đây là đúng giá trị của nghiệm θ ≈ 0 (residual bằng thông lượng laser), không phải lỗi đo.

Kết luận: hướng 2 riêng lẻ không đạt tiêu chí 27.24(e) trên GPU. Nghiệm nguội lặp lại ở 20 000 vòng, nên hướng này không đủ để thoát nghiệm nguội. Lượt `spotbal` (hướng 1) và kết hợp hai hướng chưa có kết quả và cần chạy tiếp.

**27.25 Hướng A: giữ thông lượng laser thật, sửa cách huấn luyện (ngày 2026-10-13, quyết định của người nghiên cứu).**

(a) Bối cảnh. Các lượt không kẹp T (27.22, 27.24) đều nguội: nhiệt cực đại 293–455 K, không có vũng. Lượt chỉ tối ưu biên trên học được (sai số xuống 5·10⁻⁵), nhưng khi thêm PDE thì nghiệm quay về trạng thái nguội. Giả thuyết cần kiểm tra: tối ưu Adam với lấy mẫu ngẫu nhiên không đủ để giữ đồng thời PDE và biên trên. Hướng này không thay đổi vật lý: vẫn Gaussian bề mặt, đối lưu và bức xạ, không bay hơi (27.23), không kẹp T, không L_E, T_amb 293 K, biên bên cách nhiệt.

(b) Bằng chứng ủng hộ hướng. Peng & Panesar 2023 (loss lệch bậc và output transform ảnh hưởng lớn); arXiv 2506.20537 (hạng PDE thấp rồi tăng, cần trọng số cố định theo bậc độ lớn); Sharma & Guo 2025 và Piandoro 2026 dùng L-BFGS sau Adam; Zhu 2021 và Faegh 2026 cho ràng buộc cứng IC và biên (không dùng cho vết laser).

(c) Các thành phần của hướng A, thử theo thứ tự:
- A1. Adam dài hơn kết hợp hai hướng 27.24 (chia biên trước, cân bằng vết laser, tăng dần PDE).
- A2. L-BFGS sau Adam, với tập điểm cố định (không lấy mẫu lại mỗi vòng). Số vòng L-BFGS và điều kiện dừng ghi trước khi chạy. Đây là phần cần thêm vào mã (`--lbfgs-iters`, `--fixed-points`), chưa có.
- A3. Mạng lớn hơn (ví dụ 5 lớp × 64 nơron) nếu A1–A2 chưa đủ. Không thử cùng lúc với A1–A2.
- A4. Trọng số thích nghi theo gradient: chỉ thử nếu A1–A3 không đạt. Không viết trong prereg này.

(d) Quy trình. Mọi thử nghiệm đầu tiên chạy trên CPU ngắn (cùng cấu hình seed 0, miền thu nhỏ như 27.24(d)) để kiểm tra mã và xu hướng. Chỉ lượt đạt tín hiệu sớm mới chạy GPU đầy đủ (20 000 vòng, seed 0). Mỗi lượt GPU ghi diagnose và profile như 27.22.

(e) Tiêu chí đạt (không đổi so với 27.24(e)): (i) nhiệt cực đại trong vết laser vượt T_sol = 1877 K; (ii) xuất hiện vũng nóng chảy có rộng và sâu lớn hơn 0; (iii) hạng biên trên trong vết laser và hạng PDE cùng nhỏ, không đánh đổi nhau. Kết quả so với Bảng 3 được báo cáo đầy đủ, không chọn lượt theo kích thước mục tiêu (27.4).

(f) Tín hiệu sớm để chuyển từ CPU sang GPU: sau tối đa 4000 vòng Adam (hoặc sau L-BFGS nếu có), nhiệt cực đại trên lưới vượt 1000 K và hạng PDE không tăng quá 10 lần so với giá trị ban đầu. Nếu không đạt tín hiệu này, lượt không chạy GPU và được báo cáo là không đạt ở giai đoạn CPU.

(g) Giới hạn đã biết. Nếu A1–A3 đều không đạt, kết luận là PINN thuần không tái tạo được thông lượng laser ở cấu hình này, và kết quả âm được báo cáo. Không dùng kẹp T hay dữ liệu mục tiêu để cứu kết quả.

**27.25 (kết quả CPU A2, ngày 2026-10-13).** Lượt 2000 vòng Adam (không hướng 27.24, seed 0, miền thu nhỏ) rồi 50 bước L-BFGS trên 1000 điểm cố định. Adam kết thúc với hạng PDE 6.5·10⁻⁶ và hạng biên trên 0.0089. L-BFGS không tiến triển: loss 0.010334 ở bước đầu và giữ nguyên ở các bước sau, tức là đường tìm kiếm không giảm được loss. Diagnose: nhiệt cực đại 295.9 K, bề mặt 295.5 K, không vũng, sai số năng lượng 0.991. Không đạt tiêu chí 27.25(e). Kết quả cho thấy L-BFGS trên tập điểm cố định không thoát được nghiệm nguội trong cấu hình này.

**27.25 (h) Hướng A4: trọng số thích nghi theo độ lớn gradient (ngày 2026-10-13, quyết định của người nghiên cứu).** Cờ `--adaptive-weights`, mặc định tắt. Mỗi `--adapt-every` vòng (mặc định 100), tính ‖∇θ L_k‖ cho ba hạng k ∈ {PDE, biên trên, biên bên}, rồi đặt w_k = max_j ‖∇L_j‖ / ‖∇L_k‖, làm trơn bằng EMA với hệ số `--adapt-alpha` (0.9). Hạng PDE được giữ ở trọng số tham chiếu nếu cần; trọng số PDE vẫn nhân với ramp của 27.24(b) nếu có. Cơ sở: Wang, Teng và Perdikaris (2021). Không đổi vật lý, không kẹp T, không dùng dữ liệu mục tiêu (27.4).

Kiểm tra mã: lượt 300 vòng CPU cho w_top ≈ 574 và w_face ≈ 1032, w_pde = 1; loss cuối 4.68 do hạng biên trên nhân trọng số lớn. Hệ số biên bên lớn vì hạng biên bên rất nhỏ (~10⁻⁷). Đây chưa phải kết quả.

Tiêu chí và quy trình như 27.25(e)–(g): thử CPU ngắn 2000 vòng trước, chỉ chạy GPU nếu đạt tín hiệu sớm.

**27.25 (h, kết quả CPU A4, ngày 2026-10-13).** Lượt 2000 vòng Adam với `--adaptive-weights` (seed 0, miền thu nhỏ). Trọng số cuối: w_pde = 1, w_top ≈ 153, w_face ≈ 526. Hạng PDE cuối 9.8·10⁻⁵, hạng biên trên cuối 0.0089. Diagnose: nhiệt cực đại trên lưới 449 K, bề mặt 326 K, không vũng, sai số năng lượng 0.853. Không đạt tiêu chí 27.25(e). Trọng số thích nghi làm nhiệt cực đại tăng từ ~330 K (đối chứng cùng cỡ) lên 449 K nhưng chưa đủ để vượt T_sol hay tạo vũng.

**27.25 (i) Hướng A3: mạng lớn hơn (ngày 2026-10-13, quyết định của người nghiên cứu).** Tham số mới `--width` (mặc định 64) và `--depth` (mặc định 4, số lớp ẩn). A3 dùng 5 lớp ẩn × 64 nơron. Diagnose và profile đọc kiến trúc từ checkpoint, nên không cần truyền lại tham số. Mã đã kiểm tra: test 9/9 pass, huấn luyện 30 vòng và diagnose chạy được với depth 5.

Lệnh GPU (seed 0, cấu hình chính, 20 000 vòng, lr 10⁻³ → 10⁻⁴ trong 5000 vòng cuối, không kẹp T, không bay hơi, T_amb 293 K, biên bên cách nhiệt, mạng 5 × 64):
```
python -m lpbfpinn.r2022.train --device cuda --surface flux --side-bc adiabatic --seed 0 --iters 20000 --lr 1e-3 --lr-final 1e-4 --decay-iters 5000 --depth 5 --width 64 --out results/pinn_r2022_flux_A3
```
Sau khi chạy, diagnose và profile trên `results/pinn_r2022_flux_A3/model.pt` như 27.22. Tiêu chí như 27.25(e), không đổi (27.4).

**27.25 (j) Hướng A5: nhúng Fourier cho đầu vào không gian-thời gian (ngày 2026-10-13, quyết định của người nghiên cứu; theo Farrag và cộng sự 2025, mục 4.4 về độ lệch phổ).** Cờ `--fourier N` (mặc định 0 = tắt), `--fourier-sigma σ` (mặc định 10). Đầu vào đã chuẩn hóa [−1, 1] được chiếu lên N tần số ngẫu nhiên cố định B ~ N(0, σ²) (4 × N, seed 12345, không phụ thuộc seed huấn luyện), rồi nối [x, sin(2πxB), cos(2πxB)] trước mạng tanh. Không đổi vật lý, không kẹp T, không bay hơi, không dữ liệu mục tiêu (27.4). Ràng buộc cứng IC và đáy giữ nguyên vì chỉ thay đầu vào.

Mã đã kiểm tra: test 9/9 pass; huấn luyện 30 vòng với N = 32 và diagnose đều chạy được. Diagnose và profile đọc N và σ từ checkpoint.

Thử CPU so với đối chứng cùng cấu hình (2000 vòng Adam, seed 0, miền thu nhỏ, cùng lệnh như lượt cơ bản 27.24(d)): N = 32, σ = 10. Tiêu chí sớm để chạy GPU: nhiệt cực đại vượt 1000 K và hạng PDE không tăng quá 10 lần so với đối chứng. Tiêu chí đạt như 27.25(e). Nếu không đạt tín hiệu sớm, không chạy GPU và báo cáo là không đạt ở giai đoạn CPU.

**27.25 (j, kết quả CPU A5, ngày 2026-10-13).** Lượt 2000 vòng Adam với N = 32, σ = 10 (cùng cấu hình đối chứng). Hạng PDE ban đầu 912, cuối 0.497 (đối chứng 7·10⁻⁶). Hạng biên trên cuối 0.0088. Diagnose: nhiệt cực đại trên lưới 9302 K, bề mặt 9295 K, vũng rộng 385 µm và sâu 200 µm, sai số năng lượng −25.2. Không đạt tín hiệu sớm (PDE tăng hơn 10 lần so với đối chứng và nhiệt vượt xa giới hạn vật lý). Không chạy GPU. Nguyên nhân khả dĩ: σ = 10 tạo tần số quá cao so với kích thước miền, khiến PDE không hội tụ và mạng dao động quanh nghiệm rất nóng. Chưa thử σ nhỏ hơn; nếu muốn tiếp tục A5, cần một quét σ trên CPU (ví dụ 1 và 3) có ghi trước trong prereg.

**27.25 (j2) Quét σ cho nhúng Fourier (ngày 2026-10-13, quyết định của người nghiên cứu).** Sau 27.25(j) (σ = 10 không đạt), quét σ ∈ {1, 3} với N = 32, cùng cấu hình CPU 2000 vòng Adam, seed 0. Tiêu chí sớm để chạy GPU, ghi trước khi chạy: (i) nhiệt cực đại trên lưới trong khoảng 1000–4000 K; (ii) hạng PDE cuối không vượt 10 lần đối chứng (7·10⁻⁵); (iii) không có vũng vượt Bảng 3 đáng kể. Đạt cả ba thì mới đề xuất chạy GPU. Không chọn σ theo kích thước mục tiêu (27.4).

**27.25 (j2, kết quả quét σ trên CPU, ngày 2026-10-13).** Cùng cấu hình CPU 2000 vòng, N = 32.
- σ = 1: hạng PDE cuối 1.6·10⁻⁵ (đạt tiêu chí ii), nhiệt cực đại 352 K (không đạt tiêu chí i), không vũng.
- σ = 3: hạng PDE cuối 5.6·10⁻⁴ (không đạt tiêu chí ii), nhiệt cực đại 590 K (không đạt tiêu chí i), không vũng.
Không σ nào đạt tiêu chí sớm. Không chạy GPU. Với σ nhỏ, PDE ổn định nhưng nghiệm vẫn nguội; với σ lớn hơn, nhiệt tăng nhưng PDE không hội tụ. Nhúng Fourier đơn thuần không giải quyết vấn đề nguội trong phạm vi σ đã quét.

**27.26 Mốc kiểm tra theo giai đoạn: Rosenthal và cân bằng năng lượng trước (ngày 2026-10-13, quyết định của người nghiên cứu).**

(a) Hai cổng đầu tiên cho giai đoạn 1 (PINN dẫn nhiệt thuần, Ti6Al4V không bột, nguồn Gaussian di chuyển):
- **M1a, cân bằng năng lượng:** sai số tương đối (E_in − E_ra − E_tích)/E_in không vượt 5%, đo bởi `diagnose` (`energy.rel_error`).
- **M1c, Rosenthal:** độ rộng vũng trên mặt tại T_sol trong khoảng ±20% của giá trị Rosenthal, tính bởi `lpbfpinn.r2022.rosenthal` với cùng P hấp thụ, v, k, ρ, cp, T_sol, T₀. Với hằng số hiện tại: rộng 106 µm, khoảng chấp nhận 85–127 µm. Độ sâu Rosenthal 53 µm được báo cáo nhưng không là cổng, vì nguồn điểm không mô tả đúng độ sâu gần vết laser.

(b) Cổng M0 (kiểm tra toán) đã đạt, gồm test PDE trên nghiệm giải tích và biên Robin. M1b (không nguội: nhiệt cực đại vượt T_sol và có vũng) vẫn là điều kiện trước M1c, vì M1c chỉ có ý nghĩa khi vũng tồn tại.

(c) Giới hạn của Rosenthal: nguồn điểm, bán không gian, quasi-steady, không có bay hơi hay phản lực, không có kích thước vết laser 50 µm. Nó chỉ là mốc tham chiếu cho giai đoạn dẫn nhiệt, không phải mục tiêu Bảng 3. Hằng số k = 33.1 W/m·K chưa xác nhận với Hình 14 của Ransenigo; độ rộng gần như không đổi khi k giảm xuống 7 (106 µm so với 108 µm), nên cổng M1c ít nhạy với k.

(d) Đánh giá các lượt đã có theo hai cổng (không chạy thêm):
- Kẹp T, lr giảm, GPU 20 000 vòng: năng lượng 0.437 (không đạt M1a); rộng 165 µm, ngoài khoảng 85–127 (không đạt M1c); nhiệt cực đại 3543 K (vượt T_sol, nhưng vũng sai). Đạt M1b, không đạt M1a và M1c.
- Thông lượng không kẹp (pderamp GPU, các lượt CPU): năng lượng 0.85–0.998, nhiệt không vượt 500 K. Không đạt M1a, M1b, M1c.

(e) Quy trình: giai đoạn 1 chỉ chuyển sang giai đoạn 2 (bột) khi M1a, M1b và M1c cùng đạt trên cùng một lượt chạy có seed đã ghi. Nếu không đạt, báo cáo là không đạt giai đoạn 1 và không thêm bột để che lỗi.

**27.26 Giai đoạn 1: cấu trúc bài toán PINN thuần (Ti6Al4V, không bột) (ngày 2026-10-13, đề xuất để người nghiên cứu xác nhận).**

(a) Phạm vi. Khối Ti6Al4V đặc, dẫn nhiệt thuần (không bột, không bay hơi, không áp suất phản lực, không đối lưu trong vũng, không nhiệt ẩn). Cùng phạm vi với nghiệm Rosenthal và với Faegh và cộng sự 2026, nên các so sánh là nhất quán. Bột và nhiệt ẩn là giai đoạn 2 và 1b.

(b) Phương trình (thứ nguyên). ρc ∂T/∂t = ∇·(k∇T) trong miền, với ρ, c, k hằng số (giả định đơn giản hóa, cần đối chiếu Hình 14 của Ransenigo; xem (f)).

(c) Điều kiện ban đầu và biên.
- Ban đầu: T(x, 0) = T_amb = 293 K.
- Đáy z = 0: T = T_amb (Dirichlet; Ransenigo Hình 5, Faegh, Yuan).
- Mặt trên z = L_z, nguồn di động: k∂T/∂z = Q(x,y,t) − [h(T − T_amb) + εσ(T⁴ − T_amb⁴)], với Q = 2αP/(πr_b²)·exp(−2ρ²/r_b²) di chuyển theo x_l = x_start + vt, có đoạn khởi động. Dấu theo pháp tuyến ngoài (đã kiểm chứng cùng Zhu eq. 8 và Yuan eq. 2b). h = 25, ε = 0.2 là giá trị hiện tại, cần lý do (f).
- Mặt bên: cách nhiệt (mặc định); Robin là lựa chọn đối chứng (27.15).

(d) Ràng buộc: IC và đáy là ràng buộc cứng θ = z·softplus(raw)·(1 − e^{−t}) (Zhu 2021 cho ràng buộc cứng; dạng cụ thể là thiết kế riêng). Mặt trên và mặt bên là hạng mất mát mềm. Không kẹp T, không dùng dữ liệu thực nghiệm.

(e) Chiến lược huấn luyện. Adam, lr 10⁻³ giảm tuyến tính đến 10⁻⁴ trong 5000 vòng cuối, tổng 20 000 vòng (GPU), seed 0. Lấy mẫu: miền đều và Sobol, cộng điểm quanh vết laser; mặt trên lấy mẫu gồm điểm gần laser và điểm xa (y ≥ y_tip + 100 µm). Loss = w_pde·L_PDE + w_top·L_top + w_face·L_face, chuẩn hóa theo thông lượng đỉnh và theo nhiệt độ. Không dùng các tùy chọn đã thử ở 27.24, 27.25 (chia pha, trọng số thích nghi, L-BFGS, Fourier) làm cấu hình mặc định; chỉ dùng làm đối chứng có nhãn.

(f) Các giả định đơn giản hóa cần xác nhận.
- ρ = 4106 kg/m³: Ransenigo ghi 4430 kg/m³ ở nhiệt độ phòng. Cần nguồn.
- k = 33.1 W/m·K: có vẻ cao so với Ti6Al4V. Cần đối chiếu Hình 14.
- h = 25, ε = 0.2 (từ FV): khác Yuan, Faegh, Piandoro. Cần lý do hoặc đổi.
- Vật liệu hằng số: khác Ransenigo và Yuan (phụ thuộc T). Ghi là hạn chế.

(g) Mốc đánh giá: đã bỏ theo quyết định của người nghiên cứu (không còn M0–M2 trong prereg).

(h) Việc cần làm theo thứ tự: (1) viết mốc Rosenthal vào diagnose hoặc script riêng; (2) xác nhận vật liệu (f); (3) sửa cân bằng năng lượng (M1a) rồi mới đánh giá M1b, M1c. Không thêm bột trước khi M1 đạt.

(i) Đối chiếu thông số vật liệu (ngày 2026-10-13). Hình 14 của Ransenigo và cộng sự 2022 (ảnh, đọc bằng mắt, sai số khoảng ±5%):
- ρ: khoảng 4440 kg/m³ ở 300 K, giảm xuống khoảng 4200 ở 1850 K và 4000 ở 2100 K. Giá trị 4106 của mã tương ứng nhiệt độ gần 1900 K, không phải nhiệt độ phòng.
- c_p: khoảng 600 J/kg·K ở 300 K, tăng lên khoảng 800 ở 1800 K, có đỉnh khoảng 1200 quanh 1100–1300 K. Giá trị 838 của mã tương ứng nhiệt độ gần 1900 K.
- k: khoảng 7 W/m·K ở 300 K, khoảng 16 ở 900 K, khoảng 25 ở 1300 K, khoảng 33–35 ở 1900–2000 K. Giá trị 33.1 của mã tương ứng nhiệt độ gần 1900 K.
Kết luận: ba hằng số hiện tại là giá trị gần điểm nóng chảy. Dùng hằng số cho cả miền làm sai nhiệt độ vùng xa vết laser (k thực tế thấp hơn nhiều ở nhiệt độ phòng). Đây là hạn chế cần ghi, hoặc cần k(T), c_p(T), ρ(T) theo Hình 14.

Tìm kiếm bài khác (chưa mở toàn văn, chỉ dựa trên tóm tắt tìm kiếm): (1) preprint Research Square rs-10433902 có công thức k, c_p, ρ theo nhiệt độ cho Ti-6Al-4V, cần kiểm chứng đã bình duyệt chưa; (2) Bartsch et al., Frontiers in Mechanical Engineering 2022 (fmech.2022.830104): đo độ dẫn nhiệt Ti-6Al-4V trong LPBF, khối và bột; (3) arXiv 2604.24669 (mô hình 1D, Ti-6Al-4V): bảng tham số, ρ ≈ 4090, c ≈ 1130, k ≈ 20, cần kiểm tra lại vì cột bảng không rõ ràng. Hệ số phát xạ đo được 0.36 (một nghiên cứu hiệu chuẩn IR, không phải bột LPBF). Không tìm được bảng chuẩn cho h và ε trong nung Ti-6Al-4V bằng laser.

**27.27 Đặc tả cuối cùng cho PINN thuần giai đoạn 1 (ngày 2026-10-13, quyết định của người nghiên cứu D1–D8 theo mục 27.26).**

(a) Phạm vi: khối Ti-6Al-4V đặc, không bột, không bay hơi, không áp suất phản lực, không dòng chảy trong vũng, không nhiệt ẩn. Dẫn nhiệt không dừng với nguồn Gaussian bề mặt di chuyển, cách nhiệt hai mặt bên.

(b) D1, kích thước miền: giữ nguyên kích thước mặc định ban đầu (x ∈ [−750, 750] µm, y ∈ [0, 380] µm, z ∈ [0, 240] µm). Ghi nhận rủi ro: với z_max = 240 µm, nhiệt có thể ảnh hưởng đến đáy trong thời gian quét. Không sửa theo quyết định này.

(c) D2, vật liệu theo Ti-6Al-4V (hằng số, không phụ thuộc nhiệt độ): ρ = 4122 kg/m³, c = 831 J/kg·K, k = 35 W/m·K (Sharma và cộng sự 2026, Bảng 2). Thay thế ρ = 4106, c = 838, k = 33.1 của cấu hình cũ. Phụ thuộc nhiệt độ theo Han và cộng sự 2025 (Bảng 1) là lựa chọn nâng cấp về sau, không dùng trong giai đoạn 1.

(d) D3, đối lưu và bức xạ: h = 25 W/m²·K, ε = 0.2 (giữ giá trị cũ theo quyết định của người nghiên cứu). Ghi nhận: Sharma 2026 dùng ε = 0.4, Han 2025 dùng h = 40, ε = 0.7; khác biệt này được ghi là giới hạn.

(e) D4, nhiệt đáy và nhiệt ban đầu: 298 K (theo Sharma 2026, Piandoro 2026, Ransenigo Hình 5). Nhiệt môi trường T_amb = 298 K cũng dùng cho biên đối lưu, bức xạ và điều kiện ban đầu, để giữ một giá trị duy nhất. Điều này thay thế lựa chọn 293 K trước đây; cần xác nhận lại (xem mục (k)).

(f) D5, biên bên: cách nhiệt (∂θ/∂n = 0).

(g) D6, chuẩn hóa hàm loss: mọi hạng residual chia cho θ_peak theo đơn vị không thứ nguyên, tương đương θ_peak/t_c với hạng PDE và θ_peak/L_c với hạng biên. θ_peak là nhiệt độ đỉnh kỳ vọng của trường không thứ nguyên, cố định trước khi chạy và ghi trong checkpoint. Hệ quả: hạng biên trên không còn chia cho đỉnh thông lượng như trước, nên trọng số hiệu dụng của biên trên thay đổi; cần ghi khi báo cáo.

(h) D7, learning rate: 10⁻³ giảm tuyến tính đến 10⁻⁴ trong 25% vòng cuối (5000 trên 20 000 vòng).

(i) D8, tiêu chí: không có mốc đạt/không đạt ở giai đoạn 1. Mỗi lượt báo cáo đầy đủ các chỉ số trong 27.26(7) (cân bằng năng lượng, nhiệt cực đại, vũng tại T_sol, so với Rosenthal) mà không gán nhãn.

(j) Cấu hình huấn luyện: Adam, 20 000 vòng trên GPU, seed 0 (sau đó seed 1, 2 để đo lặp lại), lấy mẫu đều và Sobol quanh vết laser. Không dùng các tùy chọn đã thử (biên trước, cân bằng vết laser, tăng dần PDE, L-BFGS, Fourier, trọng số thích nghi) làm mặc định.

(k) Cần xác nhận: (1) D2 được đọc là hằng số Ti-6Al-4V của Sharma 2026; (2) D4 thay thế T_amb = 293 K bằng 298 K cho toàn bộ. Nếu không đúng ý, ghi lại và cập nhật prereg.

(l) Việc tiếp theo: triển khai bulk (cờ powder = False), chuẩn hóa thống nhất theo θ_peak, cập nhật vật liệu, rồi kiểm thử đơn vị và chạy CPU ngắn để kiểm tra mã (không thành mốc).

**27.27 Đặc tả PINN thuần giai đoạn 1 (Ti6Al4V, không bột) (ngày 2026-10-13, quyết định của người nghiên cứu D1–D8 theo mục 27.26).**

- **D1 Kích thước miền:** z_max = 600 µm. x và y giữ như cấu hình cũ (x ∈ [−750, 750] µm, y ∈ [0, 380] µm).
- **D2 Vật liệu:** hằng số theo Sharma, Raissi, Guo 2026 (Procedia CIRP 138): ρ = 4122 kg/m³, c = 831 J/kg·K, k = 35 W/m·K. Không dùng phụ thuộc nhiệt độ ở giai đoạn này.
- **D3 Đối lưu và bức xạ:** h = 25 W/m²·K và ε = 0.2 (giữ như cấu hình hiện tại theo quyết định của người nghiên cứu). Ghi chú: các bài đã tải lên dùng ε = 0.4 (Sharma 2026), ε = 0.7 (Han 2025) và h = 40 (Han 2025). Đây là giới hạn cần nêu trong báo cáo.
- **D4 Nhiệt đáy:** T = 298 K (Dirichlet), như Sharma 2026 và Ransenigo Hình 5. Ban đầu T = 298 K.
- **D5 Biên bên:** cách nhiệt (∂T/∂n = 0).
- **D6 Chuẩn hóa loss:** thống nhất theo θ_peak/t_c. Cụ thể hóa các hệ số trong bước triển khai, có kiểm thử đơn vị trước khi chạy.
- **D7 Learning rate:** 10⁻³, giảm tuyến tính về 10⁻⁴ trong 25% vòng cuối.
- **D8 Tiêu chí dừng giai đoạn 1:** không có mốc đạt/không đạt. Mọi lượt được báo cáo đầy đủ theo mục 27.27(đầu ra) mà không chọn lượt theo kích thước mục tiêu (27.4).

Đầu ra mỗi lượt: cân bằng năng lượng (E_in, E_out theo biên, E_stored, rel_error), nhiệt cực đại trên lưới, nhiệt bề mặt, nhiệt tại 50 µm dưới tâm, profile trên mặt, vũng tại T_sol (rộng, sâu), và so sánh với Rosenthal ở vùng xa.

**27.27 Đặc tả PINN thuần giai đoạn 1 (ngày 2026-10-13, quyết định của người nghiên cứu D1–D8).**

(a) Miền: x ∈ [−750, 750] µm, y ∈ [0, 380] µm (như cấu hình cũ), z ∈ [0, 600] µm (D1). Không có bột (κ = 1, g = 1).

(b) Vật liệu hằng số theo Sharma và cộng sự 2026 (D2): ρ = 4122 kg/m³, c_p = 831 J/kg·K, k = 35 W/m·K. Không nhiệt ẩn, không bay hơi (cùng Sharma 2026).

(c) Biên: đáy T = 298 K, Dirichlet (D4); biên bên cách nhiệt (D5); ban đầu T = 298 K. Mặt trên: k∂T/∂z = Q − [h(T − T_amb) + εσ(T⁴ − T_amb⁴]). ε = 0.4 theo Sharma 2026 (D3, phần ε). Giá trị h: Sharma 2026 không nêu (chỉ có ε). Các bài khác: Han 2025 h = 40, Yuan 2025 và Faegh 2026 h = 20, FV h = 25. **Chưa chọn h. Đề xuất h = 25 (giữ từ FV), kiểm tra độ nhạy h ∈ {20, 40} sau.**

(d) Nguồn: Gaussian bề mặt, αP = 0.4 × 300 W, r_b = 50 µm, v = 1.75 m/s, đoạn khởi động 10 µm.

(e) Ràng buộc cứng: IC và đáy (θ = z·softplus(NN)·(1 − e^{−t})). Mặt trên và mặt bên là hạng mất mềm.

(f) Hàm loss chuẩn hóa thống nhất theo θ_peak/t_c (D6): mọi residual đưa về cùng đơn vị θ/t_c trước khi bình phương. Trọng số w_PDE = w_top = w_side = 1 ban đầu.

(g) Huấn luyện: Adam, lr = 10⁻³ (D7), giảm tuyến tính về 10⁻⁴ trong 25% vòng cuối, 20 000 vòng, seed 0 (GPU). Lấy mẫu như cấu hình cơ bản.

(h) Tiêu chí dừng giai đoạn 1: không có mốc đạt/không đạt (D8). Báo cáo đầy đủ các chỉ số ở mục 27.26(7), không chọn lượt theo kích thước mục tiêu (27.4).

(i) Xác nhận h = 25 W/m²·K (ngày 2026-10-13, quyết định của người nghiên cứu). Sharma 2026 không nêu h; giá trị giữ từ FV.

(j) Mã cho PINN giai đoạn 1 được thiết kế riêng, không dùng chung với code r2022 (quyết định của người nghiên cứu). Không thay đổi r2022. Các công thức và điều kiện biên phải được người nghiên cứu xác nhận trước khi viết mã.

**27.28 Mã PINN thuần giai đoạn 1 (ngày 2026-10-13, quyết định C1–C6 của người nghiên cứu).**

(a) Vị trí mã: `lpbfpinn/bulk/` (case, operator, train), test `lpbfpinn/tests/test_bulk.py`. Gói không import từ `lpbfpinn.r2022`; mã r2022 không đổi.

(b) Giá trị xác nhận: T_scale = 2000 K (C2). Đường quét y_tip = 190 µm, tâm miền (C3). x từ −550 đến 650 µm (C4). Thang chuẩn hóa residual θ_ref = (T_sol − T_amb)/T_scale = 0.7895 (C1).

(c) Chuẩn hóa: mỗi residual không thứ nguyên chia cho θ_ref. Lý do: PDE dạng thứ nguyên ρc·∂T/∂t − ∇·(k∇T) có thang ρcθ_refT_scale/t_c, và biên trên có thang kθ_refT_scale/L_c; sau khi chia cho các thang này, kết quả đều bằng residual không thứ nguyên chia cho θ_ref. Vậy tất cả hạng cùng thang (D6).

(d) Kiểm tra trước GPU (C6): chỉ test PDE trên nghiệm giải tích θ = z*²t*, kết quả r = z*² − 2t*. Pass.

(e) Chưa làm: diagnose (năng lượng, vũng nóng chảy), huấn luyện thử, và so sánh với Rosenthal. Sẽ bổ sung sau khi bạn xác nhận kế hoạch chạy GPU.

**27.28 (f) Huấn luyện Adam rồi L-BFGS cho PINN thuần (ngày 2026-10-13, đề xuất của người nghiên cứu).** Thêm `--lbfgs-iters N` vào `lpbfpinn/bulk/train.py` (mặc định 0). Sau Adam (lr 10⁻³ → 10⁻⁴ theo D7, 20 000 vòng), chạy N bước L-BFGS (tối đa 20 lần lặp mỗi bước, strong Wolfe) trên một tập điểm cố định, cùng hàm loss. Cơ sở: Sharma và cộng sự 2026 dùng Adam rồi L-BFGS cho cùng bài toán Ti6Al4V; các thử nghiệm trên cấu hình r2022 (27.25 A2) cho thấy L-BFGS dừng sớm nếu lấy mẫu không cố định, nên dùng tập điểm cố định.

Kiểm tra CPU: 300 vòng Adam + 5 bước L-BFGS chạy được, loss từ 4.53 xuống 3.01 qua 5 bước. Đây chỉ là kiểm tra mã.

Lệnh GPU (chưa chạy): `python -m lpbfpinn.bulk.train --device cuda --iters 20000 --lbfgs-iters 100 --out results/bulk_pinn_adam_lbfgs_s0`. Tiêu chí: như 27.27(h), không có mốc đạt/không đạt.

**27.28 (ghi chú nguồn C2, ngày 2026-10-13).** T_scale = 2000 K là lựa chọn thiết kế của người nghiên cứu, không lấy từ bài báo nào đã tải lên (đã kiểm tra: không bài nào nêu thang không thứ nguyên của nhiệt độ). Lý do chọn: làm θ ≈ 1 ở vùng nóng chảy để điều kiện tối ưu tốt hơn. T_scale chỉ đổi cách biểu diễn không thứ nguyên, không đổi nghiệm vật lý. Lượt r2022 cũ dùng 4000 K từ cấu hình trước đó.

**27.28 (g) Quét trọng số biên trên trên CPU (ngày 2026-10-13, theo đề xuất đã được người nghiên cứu đồng ý).** Thử w_top ∈ {1, 10, 100}, với w_PDE = 1 và w_side = 1. Cấu hình: 2000 vòng Adam, lr 10⁻³ → 10⁻⁴ (25% vòng cuối), miền và số điểm như kiểm tra CPU trước, seed 0.

Tiêu chí chọn, ghi trước khi chạy: đo tỷ số L_top/L_PDE tại vòng cuối. Cân bằng nếu tỷ số nằm trong [0.1, 10]. Không chọn trọng số theo nhiệt cực đại hay kích thước vũng (27.4). Nếu nhiều giá trị đạt, chọn giá trị có L_PDE nhỏ nhất. Nếu không giá trị nào đạt, báo cáo không có trọng số cân bằng trong tập đã thử.

**27.28 (h) Chẩn đoán CPU của model w_top = 1 (ngày 2026-10-13).** Viết `lpbfpinn/bulk/diagnose.py`: nhiệt trên lưới, vũng nóng chảy trên mặt theo thời gian (rộng, sâu tại tâm laser), cân bằng năng lượng (E_in, mất nhiệt trên và đáy, tích lũy). Không có ngưỡng đạt/không đạt (27.27(h)).

Kết quả (2000 vòng Adam, w_top = 1): nhiệt cực đại 11 657 K (lưới 20 µm) và 12 844 K (lưới 10 µm), vượt xa điểm sôi 3533 K. Vũng rộng 140 µm, sâu 20 µm (một đến hai ô lưới, không đáng tin). Cân bằng năng lượng: E_in = 0.082 J, E_out_top = 0.0004 J, E_out_bottom = 0.0005 J, E_stored = 0.18 J, sai số −1.24 (−1.20 với lưới 10 µm). Năng lượng tích lũy lớn hơn năng lượng vào, nên model vi phạm bảo toàn năng lượng.

Kết luận: tiêu chí cân bằng loss (27.28(g)) không đảm bảo nghiệm vật lý. Không chạy GPU với cấu hình này. Cần sửa trước: nguyên nhân có thể là hạng biên trên không ràng buộc đủ nhiệt độ bề mặt, hoặc thiếu chuẩn hóa theo thang nhiệt độ. Đối chiếu: Wu và cộng sự 2018 cho nhiệt cực đại 5439 K không bay hơi, nên 11 657 K là quá cao.

**27.28 (i) Kiểm tra hướng 1: biên trên có được thỏa không (ngày 2026-10-13).** Trên model w_top = 1, đo residual biên trên (dimensional, chia cho đỉnh thông lượng): tại vết laser rms = 5.7%, xa vết laser rms = 0.9%. Tại nhiệt 8277 K, mất nhiệt đối lưu và bức xạ chỉ bằng 0.35% thông lượng đỉnh.

Kết luận: biên trên không được thỏa hoàn toàn (5.7% tại vết laser), nhưng ngay cả khi thỏa tốt hơn, mất nhiệt đối lưu và bức xạ không đủ để giới hạn nhiệt bề mặt ở vài nghìn K. Nhiệt cao vì mô hình không có cơ chế giới hạn nhiệt bề mặt (bay hơi, mục 27.21 và 27.23 đã bỏ). Vì vậy hướng 1 không giải quyết được vấn đề nhiệt cực đại. Sai số năng lượng −1.2 cần kiểm tra thêm; 5.7% sai số biên không đủ giải thích hoàn toàn.

**27.28 (j) Nguyên nhân sai số năng lượng của model w_top = 1 (ngày 2026-10-13).** Tách cân bằng năng lượng theo thời gian (12 mốc, lưới 20 µm). Ở t* = 0.25, năng lượng tích lũy đã là 0.037 J trong khi năng lượng vào chỉ 0.007 J; mất nhiệt qua biên trên, đáy, và hai mặt bên đều nhỏ (dưới 0.003 J). Sai số tăng đều từ −0.03 đến −0.10 J, nên nguồn chính là nhiệt tích lũy trong miền chứ không phải thông lượng biên.

Nguyên nhân: vùng xa laser bị nóng sớm. Tại t* = 0.25, điểm cách laser 750 µm có T ≈ 385 K ở mặt trên, trong khi nhiệt chỉ lan khoảng 40 µm. Cơ chế: dạng nghiệm θ = z·softplus(raw)·(1 − e^{−t}) không thể cho θ = 0 ở vùng xa, vì softplus luôn dương (sàn ≈ 0.007 khi raw ≈ −5). Nhân với z* = 12 ở mặt trên, sàn này thành θ ≈ 0.04. Đây là lỗi cấu trúc của ràng buộc cứng IC, không phải lỗi huấn luyện hay lưới.

Hệ quả: tất cả các số đã đo trên model w_top = 1 (nhiệt cực đại, vũng, sai số năng lượng) đều bị ảnh hưởng bởi sàn nhiệt này. Cần sửa dạng nghiệm trước khi đánh giá lại. Các lựa chọn sửa (chưa thực hiện): (a) thay softplus bằng hàm có thể bằng 0 đúng, ví dụ raw²; (b) bỏ ràng buộc cứng IC và dùng hạng mất mềm cho IC; (c) trừ sàn bằng một hàm đã biết có cùng dạng. Cần xác nhận lựa chọn trước khi sửa.

**27.28 (k) Kết quả sau khi sửa dạng nghiệm raw² (CPU 2000 vòng Adam, ngày 2026-10-13).**
- w_top = 1: cân bằng loss đạt (tỷ số biên trên/PDE = 3.18). Sai số năng lượng −2.7%, giảm mạnh so với −124% trước khi sửa, xác nhận nguyên nhân là sàn nhiệt của dạng softplus. Nhiệt cực đại 12 623 K, vẫn quá cao so với điểm sôi 3533 K. Vũng rộng 100 µm, sâu 20 µm (một ô lưới, không đáng tin).
- w_top = 10: cân bằng loss đạt (tỷ số 0.42) nhưng sai số năng lượng −54%.
- w_top = 100: không đạt cân bằng (tỷ số 0.06), sai số năng lượng −296%, vũng sâu 480 µm (bất thường).
Theo tiêu chí 27.28(g), chọn w_top = 1. Vấn đề nhiệt cực đại vẫn còn và được giải thích bởi việc không có bay hơi (27.28(i)).

**27.28 (l) Kiểm tra độ sâu vũng với lưới 10 µm (model w_top = 1, dạng raw², ngày 2026-10-13).** Lưới 10 µm: nhiệt cực đại 13 577 K, vũng rộng 120 µm, sâu 20 µm, sai số năng lượng −0.5%. Lưới 20 µm: rộng 100 µm, sâu 20 µm, sai số năng lượng −2.7%. Độ sâu 20 µm giữ nguyên ở cả hai lưới và ở mọi thời điểm từ đầu lượt, nên đây là kết quả của model, không phải lỗi lưới. Độ rộng phụ thuộc lưới (100–120 µm). So với mục tiêu Bảng 3 (rộng 84.4 ± 12.6 µm, sâu 68.0 ± 10.0 µm), độ sâu lệch khoảng 3 lần; độ rộng vượt mục tiêu khoảng 20–40%. Cân bằng năng lượng tốt ở lưới mịn.

**27.28 (m) So sánh dạng Zhu (H_z·H_t) với dạng raw² (CPU 2000 vòng, w_top = 1, ngày 2026-10-13).**
- Dạng Zhu: hạng biên trên cuối 7.23, không khớp được thông lượng laser (tỷ số biên trên/PDE 192). Nhiệt cực đại 994 K, không có vũng. Sai số năng lượng +2.19 (sai dấu). Nhiệt tại điểm xa laser (x = −750 µm, t* = 0.25) là 367 K, cao hơn 298 K.
- Dạng raw² (đối chứng): nhiệt cực đại 12 623 K, vũng 100 µm × 20 µm, sai số năng lượng −2.7%. Nhiệt tại điểm xa laser 300 K.
Kết luận: dạng Zhu không khắc phục được vấn đề, và làm xấu hơn (biên trên không học được, nhiệt xa cao hơn). Giữ dạng raw² làm cấu hình hiện tại. Lý do không dùng dạng Zhu có thể là δ_z, δ_t chọn chưa phù hợp, hoặc H_z, H_t làm mạng không có gradient tốt tại biên; chưa kiểm tra thêm.

**27.28 (n) Dạng Faegh (IC bằng output transform, đáy là hạng mất mềm), CPU 2000 vòng, w_top = 1 (ngày 2026-10-13).** So với raw²:
- Nhiệt cực đại 12 516 K (raw² 12 623 K), vũng 100 × 20 µm (giống raw²).
- Sai số năng lượng +0.26 (raw² −0.027). Năng lượng tích lũy 0.059 J so với vào 0.082 J.
- Nhiệt xa laser ở t* = 0.25: 326 K (raw² 300 K).
- Hạng đáy cuối 4·10⁻⁴ (đáy không được thỏa tuyệt đối vì là hạng mềm).
- Tỷ số biên trên/PDE 3.66.
Kết luận: dạng Faegh không cải thiện so với raw²; năng lượng và nhiệt xa laser xấu hơn. Giữ raw² làm cấu hình hiện tại.

**27.29 Sink nhiệt hóa hơi làm nhiệt dung biểu kiến, phạm vi giai đoạn 1 (ngày 2026-10-13, quyết định của người nghiên cứu).** Sửa đổi có phạm vi của 27.21 và 27.23: không đưa bay hơi Cook vào, chỉ thêm sink nhiệt hóa hơi làm nhiệt dung biểu kiến g(θ) = 1 + (L_v/c_p)·f'(T), với f(T) = ½(1 + tanh((T − T_vap)/w)), L_v = 9·10⁵ J/kg, T_vap = 3533 K, w = 100 K (lựa chọn của ta, làm trơn). Bật bằng `--latent`, mặc định tắt. Test: tích phân (g − 1) qua vùng hóa hơi bằng L_v/c_p, pass.

Kết quả CPU 2000 vòng, w_top = 1: nhiệt cực đại 3606 K (gần điểm sôi, đây là điểm đạt chính của sink). Nhưng hạng biên trên cuối 7.04 (tỷ số 107, không khớp thông lượng laser), không có vũng (sâu 0 µm, rộng 100 µm), và sai số năng lượng −2.6 với năng lượng ra qua đáy 0.030 J (lớn bất thường). Kết luận: sink giới hạn được nhiệt cực đại, nhưng làm mạng không khớp biên trên và năng lượng lệch. Cần kiểm tra lại trước khi dùng cho đánh giá.

**27.29 (b) Thử làm trơn rộng hơn, w = 300 K (CPU 2000 vòng, w_top = 1, ngày 2026-10-13).** Kết quả: nhiệt cực đại 7707 K (w = 100 K: 3606 K; raw² không sink: 12 623 K). Không có vũng (sâu 0 µm). Hạng PDE cuối 0.62 (w = 100 K: 0.066), hạng biên trên cuối 2.76 (tỷ số 4.5), sai số năng lượng +0.57 (w = 100 K: −2.6). Kết luận: làm trơn rộng hơn không khắc phục được; nhiệt vẫn vượt xa điểm sôi, PDE và biên đều xấu hơn với w = 100 K. Giữ w = 100 K làm lựa chọn tốt nhất trong các sink đã thử, nhưng sink nói chung chưa cho nghiệm đạt được.
