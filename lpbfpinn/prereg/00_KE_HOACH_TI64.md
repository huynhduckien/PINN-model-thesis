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
