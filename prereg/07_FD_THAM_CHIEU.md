# Đăng ký trước 07 — FD tham chiếu: PINN có giải đúng PDE của chính nó không?

Ngày: 2026-10-07. Bản nháp đầu 18:29 (ngưỡng theo quy ước từ nhiễu huấn luyện). **Bản này viết lại lúc 18:36 (đồng hồ), trước khi commit,** theo lựa chọn của người dùng: **tiêu chí theo hiệu ứng** (sai số PINN trên mỗi Δ phải nhỏ hơn 10% hiệu ứng được báo cáo; FD chính xác gấp khoảng 3 lần mức đó). Người dùng đã đồng ý: (1) mục đích của FD là **kiểm chứng số**; (2) bộ giải chạy trên GPU. Hệ quả của tiêu chí mới: có thể cần lưới **2.5 µm** (không còn phương án rút gọn hai lưới). **Sửa lúc 18:39 (đồng hồ), trước khi commit,** theo đề nghị của người dùng: bắt đầu từ lưới 10 µm và chỉ mịn dần khi chưa đạt, để tiết kiệm thời gian; thêm quy tắc dừng đặt trước ở mục 4.2. **Commit file này trước khi viết mã và trước khi chạy bất kỳ mốc nào.** Mọi thay đổi sau đó qua một mục sửa đổi có giờ lấy từ đồng hồ. Ngưỡng đánh dấu **[QUY ƯỚC]** là lựa chọn quy ước, không phải suy ra từ vật lý. Làm theo `CLAUDE.md`.

## 1. Câu hỏi và phạm vi

**Câu hỏi:** với bốn mô hình đã huấn luyện (vệt thường và nêm, bột như đặc và bột cách nhiệt), các Δ của PINN (nêm − vệt thường) có trùng Δ của chính PDE đó, tính độc lập bằng FD, trong mức 10% hiệu ứng không?

**Không phải câu hỏi này:** PDE có đúng vật lý không; hằng số IN718, bột Moser, k_eff, σ ẩn nhiệt, ρCp sau khi chảy có hợp lý không; nêm có làm bể chảy sâu hơn ngoài thực tế không. FD **không** sửa vật lý, **không** hiệu chỉnh, **không** dùng dữ liệu ngoài, **không** đưa vào loss của PINN.

**Luật "không FD" (`00` dòng 8, `01` dòng 11):** áp dụng cho **huấn luyện** PINN (không FD trong loss, không dữ liệu trong loss). FD ở đây chỉ là tham chiếu để đo sai số, nằm ngoài vòng huấn luyện. Nếu FD cho thấy PINN sai thì việc huấn luyện lại là quyết định riêng của người dùng, không tự động.

## 2. PDE cần giải (khóa; đối chiếu mã PINN)

Cùng nghiệm với PINN: `lpbf_pinn.pde` và `top_res` (thay bằng `maskphys.pde_masked` / `top_res_masked` ở bậc cách nhiệt), cùng miền, cùng đường quét. Số liệu theo `config.py`.

| # | Phương trình | Chi tiết khóa |
|---|---|---|
| E1 | ρC\*(y,z) ∂H/∂t = ∇·(k_s κ_s K ∇T) + q | H(T) = T + (L_f/c_p)·Φ((T−T_m)/σ); T_m = 1571 K, **σ = 150 K** (`config.SIGMA_LAT`); C\* = 1 + (0.55−1)·s(y,z), s với mép tanh **15 µm** (`iface_delta`) |
| E2a | κ_s | bột như đặc (`B3`, `B3W0`): κ_s = 1; bột cách nhiệt (`B3s`, `B3W0s`): κ_s = 1 + (r−1)·s_κ(y,z)·(1−M), r = 0.02346, s_κ có mép tanh **W = 30 µm** (`a_iface = 30/2.197 µm`), M = σ((x_front−x)/8 µm)·e(y,z), x_front = x_laser + 47 µm, e(y,z) = sigmoid(−sd/w_e), w_e = 30/4.394 µm, từ **bao T_max cố định** (mục 3.3) |
| E2b | K(T) | K = 1 + 1.8·½(1+erf((T−T_k)/(σ√2))), T_k = T_liq + 2σ = 1909 K |
| E3 | q = (2α_eff P(t)/(π r_b²))·exp(−2r²/r_b²)·β·exp(−β(L_z−z)) | α_eff = 0.422207, β = 37,414.5 m⁻¹, P(t) = P·min(t/10 µs, 1), P = 300 W, v = 0.8 m/s, r_b = 40 µm |
| E4 | mặt trên z = L_z: −k_s κ_s K ∂T/∂z = q_ev(T) | Anisimov/Fang: q_ev = L_v p_a exp(−(L_v M/R)(1/T − 1/T_b))·√(M/(2πRT)); L_v = 2.224×10⁶ J/kg, M = 0.166534 kg/mol, T_b = 3120 K, p_a = 101325 Pa; trên T_cap = 6000 K thay bằng tiếp tuyến (`q_evap`); κ_s tại mặt trên theo `kappa_static` |
| E5 | biên khác, điều kiện đầu | T = T_amb = 298 K ở đáy z = 0 và bốn mặt bên; T = T_amb tại t = 0 |
| E6 | miền, hình học | L_x = 1500 µm; L_y = 380 µm (vệt thường) hoặc 480 µm (nêm); L_z = 240 µm; lớp bột 60 µm; nêm 45°, y_a = 0; đường quét x từ −550 đến +650 µm (1.5 ms) |
| E7 | hằng số | ρ = 8000, c_p = 600, k_s = 22, T_sol = 1533, T_liq = 1609, L_f = 272 kJ/kg (không đổi theo T) |

**Một nguồn hệ số:** FD gọi các hàm của PINN (`mat_fields`, `powder_index`, `k_melt`, `latent`, `q_volume`, `q_evap`, `maskphys.kappa_static`) trên tensor lưới, không viết lại công thức. Nếu hàm nào không chạy được trên lưới GPU thì ghi vào mục sửa đổi, không chép bản riêng.

## 3. Thiết kế số học (khóa)

**3.1 Nền:** sao chép **không sửa** các hàm cần thiết từ `D:\KIEN 2026\paper\Luan van tien si\coding\scratchpad` vào `fdref.py` (ghi sha256 16 ký tự đầu của nguồn lúc sao chép: `fd_b3.py` `90a4eae5f11b8c6a`; `c1_fd_history_pilot.py` `3bbd30bcb1cb0752`; `c1_model_a_fd.py` `b300919851ab9630`): `conduction`, `boundary_loss`, `powder_properties`, `harmonic`, bảng entanpi, `interp_extent`, khung `run`. FD cũ chỉ là nguồn tham khảo và mốc hồi quy (mục 4, F0), **không** là chuẩn đúng/sai (nó gồm các mô hình bột khác vật lý với PINN).

**3.2 Sơ đồ:** thể tích hữu hạn cell-centered; độ dẫn nhiệt mặt ô theo trung bình điều hòa; Dirichlet ở mặt biên qua nửa ô; entanpi với nghịch đảo bảng; Euler tiến hiện với dt = 0.4 × giới hạn ổn định (như FD cũ); float64; GPU (RTX 4080), một tiến trình một lúc (`lock.py`). Hai thay đổi so với FD cũ (cờ `numerics = "v2"`; cờ `"old"` giữ nguyên FD cũ cho phép hồi quy):
- nguồn Beer–Lambert tích phân **chính xác theo z** trên từng ô (FD cũ dùng điểm giữa, sai ~(βh)²/24);
- mặt trên: nhiệt độ mặt T_s giải bằng Newton theo cột từ k(T_s)(T_c − T_s)/(h/2) = q_ev(T_s) (FD cũ coi bay hơi là nguồn âm trong ô trên cùng).

**3.3 Mặt nạ cách nhiệt cố định:** FD dùng đúng bao T_max mà PINN đã dùng cho lượt so sánh (mục 5), đọc từ file `envelope` và kiểm sha. Phép lặp mặt nạ (FD tự tính bao, lặp tới hội tụ) **không nằm trong 07**.

**3.4 Lưới:** đều, h = 10, 5, 2.5 µm theo **lịch thích ứng** của mục 4.2; dt tỉ lệ h².

## 4. Tiêu chí (khóa)

### 4.1 Ngân sách sai số theo hiệu ứng

Các Δ chính của PINN (nêm cách nhiệt − vệt thường, W = 30 µm; lấy từ `06` mục tham chiếu và bảng chỉ báo, **trước khi có số FD**) và ngân sách tol_Q = 0.10 × |hiệu ứng| **[QUY ƯỚC: 10%]**:

| Δ_Q (C4 − C3) | Hiệu ứng của PINN | tol_Q (10%) | Mục tiêu chính xác FD (tol_Q/3) **[QUY ƯỚC: 3]** |
|---|---|---|---|
| ΔD (sâu = độ xuyên) | +18.31 µm | 1.83 µm | 0.61 µm |
| ΔT_pk tại đáy lớp bột (50% quét) | +159.8 K | 16.0 K | 5.3 K |
| Δ thời gian lưu trên T_sol | +222.0 µs | 22.2 µs | 7.4 µs |
| tỉ số tốc độ nguội nêm/thường | 0.543 (cách 1 là 0.457) | 0.046 | 0.015 |

Thêm một Δ phụ (C2 − C1, nêm bột như đặc): **ΔD = +8.99 µm**, tol = 0.90 µm, mục tiêu FD 0.30 µm. Các đại lượng khác của C2 − C1 và mọi đại lượng riêng lẻ của từng hình học (D, rộng, T_pk, lưu của C1–C4) **chỉ ghi lại, không có phán quyết** (với bột như đặc, các số chu trình nhiệt nhỏ và đổi theo thời điểm quét, ví dụ tỉ số nguội 1.22 → 1.03 theo `05`, nên "10% hiệu ứng" vô nghĩa). Tham chiếu để đọc: nhiễu huấn luyện đo được (`06`): 0.41 µm, 10.9 K, 10 µs, 0.041, đều nhỏ hơn tol_Q tương ứng, riêng tỉ số nguội (0.041 so với 0.046) sát biên.

### 4.2 Ước sai số FD (GCI) và lịch lưới thích ứng

**Công thức.** Với lưới h₁ > h₂ > h₃ (tỉ số mịn 2): bậc quan sát p = log₂((Q₁ − Q₂)/(Q₂ − Q₃)); sai số trên lưới mịn nhất e_FD = 1.25·|Q_mịn − Q_liền trước|/(2^p − 1) (chỉ số hội tụ lưới Roache, hệ số an toàn 1.25 **[QUY ƯỚC theo thông lệ CFD; chưa đọc lại nguyên văn nguồn]**). Nếu dãy Q không đơn điệu hoặc p ngoài [0.5, 2.5], đặt e_FD = max các hiệu giữa hai lưới liền kề và ghi "bậc không xác định".

**Lịch (khóa, theo đề nghị của người dùng: bắt đầu thô, mịn dần khi chưa đạt):**
- **Giai đoạn A:** chạy C1–C4 ở 10 và 5 µm (rẻ: ước vài giây và vài chục giây mỗi lượt).
- **Giai đoạn B (luôn chạy):** chạy **C1** ở 2.5 µm (ca rẻ nhất, L_y = 380 µm) để đo bậc quan sát p_obs trên **bài toán thật** cho từng đại lượng của C1 (sâu, rộng, T_pk, lưu, tốc độ nguội). Lý do: hai lưới không đủ để biết bậc, và mép tanh 15 µm (thang 3.75 µm) cùng bay hơi có thể kéo bậc dưới 2 so với MMS.
- **Giai đoạn C:** nếu |p_obs − p_MMS| ≤ 0.5 cho mọi đại lượng của C1, tính e_FD của mọi Δ bằng **hai lưới (10, 5 µm)** với p = p_MMS (từ F2). Nếu e_FD ≤ tol_Q/3 cho mọi Δ chính và Δ phụ thì **dừng**: lưới tham chiếu là 5 µm, d_Q tính ở 5 µm.
- **Giai đoạn C′:** nếu không, chạy C2, C3, C4 ở 2.5 µm và dùng GCI ba lưới cho mọi Δ; lưới tham chiếu là 2.5 µm.

**Quy tắc chống thiên lệch (khóa):** quyết định dừng hay mịn thêm chỉ dựa vào hội tụ của chính FD (e_FD so với tol_Q/3), **không bao giờ** dựa vào việc PINN và FD có khớp hay không. `fd_compare.py` chỉ được chạy **sau khi** lưới tham chiếu đã được chọn theo quy tắc trên; không xem số PINN–FD để quyết định chạy thêm lưới.

### 4.3 Phán quyết cho từng Δ chính

Gọi d_Q = |Δ_Q(PINN) − Δ_Q(FD, lưới 2.5 µm)|.
- **ĐÚNG:** d_Q + e_FD ≤ tol_Q (ngay cả trường hợp xấu nhất trong sai số FD vẫn trong ngân sách).
- **SAI:** d_Q − e_FD > tol_Q (PINN vượt ngân sách chắc chắn), hoặc Δ_Q khác dấu.
- **KHÔNG KẾT LUẬN:** còn lại (FD chưa đủ chính xác để phân biệt).

Phán quyết cho **bảng chỉ báo** (C4 − C3): ĐÚNG nếu cả bốn Δ ĐÚNG; SAI nếu ít nhất một Δ SAI; còn lại KHÔNG KẾT LUẬN. Báo cáo ghi rõ từng Δ, và mục tiêu chính xác tol_Q/3 có đạt không.

**Hành động (khóa):** báo cáo đúng phán quyết kèm bản đồ sai số theo vùng. **Không** tự huấn luyện lại, **không** đổi ngân sách sau khi thấy số. Nếu phán quyết là SAI hoặc KHÔNG KẾT LUẬN, bảng chỉ báo ghi cả hai cột (PINN, FD) cho đến khi người dùng quyết.

### 4.4 Cổng của chính FD (trước khi chạy mục 5; một cái trượt thì dừng, báo cáo)

| Mã | Phép kiểm | Tiêu chí |
|---|---|---|
| F0 | Hồi quy (cờ `old`) tái hiện `fd_b3/F5_old_h10`, `OH0_a1s_overhang_h10` | đỉnh trong 1e-9 tương đối; rộng/sâu/dài trong 1e-6 µm |
| F1 | Nghiệm giải tích R0 (`legacy/overhang_r0.py`): vật liệu đồng nhất, không ẩn nhiệt, không bay hơi | ở 2.5 µm: sâu và rộng trong **0.61 µm** (= mục tiêu ΔD); sai số giảm đơn điệu 10 → 5 → 2.5 µm |
| F2 | MMS với toán tử PINN: nghiệm chế tạo θ_ms, số hạng nguồn bù tính bằng chính `L.pde` (và `top_res`) rồi cấp vào FD | bậc quan sát giữa 5 và 2.5 µm trong **2 ± 0.3** (lý thuyết: sơ đồ bậc hai với nghiệm trơn; ±0.3 là **[QUY ƯỚC]**; sai khác lớn nghĩa là lỗi triển khai) |
| F3 | Đóng năng lượng, mọi lượt | < 1e-9 |

## 5. So sánh PINN và FD (khóa)

Bốn ca, mỗi ca ở ba lưới, cùng điều kiện (300 W, 0.8 m/s). Checkpoint và mặt nạ lấy đúng bản đã dùng cho bảng chỉ báo:

| Ca | Mô hình | Checkpoint PINN | Mặt nạ cố định |
|---|---|---|---|
| C1 | vệt thường B3 (bột như đặc) | tag `B3_84d0289e` (`runs/M1`) | không |
| C2 | nêm B3W0 (bột như đặc, M2″) | `runs/M2pp/B3W0_ce48c3b3.pt` | không |
| C3 | vệt thường cách nhiệt B3s | `runs/Fs/B3s_it0/B3s_d2f9f1db.pt` | `runs/ENV/B3_iter0.pt` (sha 20034cb6) |
| C4 | nêm cách nhiệt B3W0s | `runs/Fs/B3W0s_it1/B3W0s_ba6419f6.pt` | `runs/Fs/B3W0s_it0/envelope_next.pt` (sha 71d24a28) |

**Đại lượng:** (i) hình học tại 70% quét bằng `indicator.pool_dims` (sâu, rộng, T_pk, độ xuyên), chung hàm cho PINN và FD; (ii) chu trình nhiệt tại (y = 0, đáy lớp bột) ở 50% quét bằng `indicator.cycle_metrics` (T_pk, thời gian lưu trên T_sol, tốc độ nguội trên [T_sol, T_liq]); (iii) Δ = nêm − vệt thường (C2 − C1; C4 − C3); (iv) **trường nhiệt** (chỉ ghi): sai số tương đối bình phương trung bình và cực đại của T_PINN so với T_FD trên vùng T_FD ≥ 600 K, tách theo vùng (gần chùm, bể chảy, đáy bể, bột dưới phần nhô, vùng xa) tại 50/70/90% quét; (v) năng lượng (chỉ ghi): E_in, E_lưu trữ, E_bay hơi, E_biên của FD cạnh E_res, E_id của PINN (`gate_light`).

## 6. Dự đoán (ghi trước, có thể sai)

- F0, F1, F2, F3 đều đạt (~70%). F1 và F2 là chỗ dễ trượt: sơ đồ mặt trên với bay hơi chưa chắc bậc hai, và mép tanh 15–30 µm có thể kéo bậc thấp hơn trong dải 5–2.5 µm.
- Giai đoạn C dừng được ở 5 µm (không cần C2–C4 ở 2.5 µm): ~20%. Lý do: mép tanh 15 µm (thang 3.75 µm) chưa được phân giải ở 5 µm và các mục tiêu chính xác chặt (ΔT_pk 5.3 K).
- Đạt mục tiêu chính xác FD (tol_Q/3) ở lưới tham chiếu (dự đoán cho trường hợp phải xuống 2.5 µm): ΔD ~50%, thời gian lưu ~40%, tỉ số nguội ~35%, **ΔT_pk ~25%** (nhiệt độ đỉnh sát mặt, nơi bay hơi chi phối).
- Phán quyết bảng chỉ báo (C4 − C3): ĐÚNG ~30%, KHÔNG KẾT LUẬN ~35%, SAI ~35%. Lý do bất định: khớp PDE của PINN cách nhiệt thô (L_PDE ~1e-2, gấp ~10 lần B3), biên E_id mỏng ở 30 µm, và ngân sách 10% khá chặt.
- Phán quyết ΔD của C2 − C1 (ngân sách 0.90 µm): KHÔNG KẾT LUẬN ~50%.
- FD hội tụ cho vệt thường B3 cho độ sâu trong 88–92 µm và rộng trong 165–169 µm (~70%).
- Nếu có Δ SAI, ΔD của FD lớn hơn của PINN (~60%). Căn cứ yếu: FD cũ cho Δ lớn hơn ở các mô hình bột khác (vật lý khác).

## 7. Quy trình thực hiện (mỗi bước có kiểm; không bước nào chạy khi bước trước trượt)

1. `fdref.py` (sao chép hàm nền, cờ `old`/`v2`) + `smoke/fd0_regress.py` → F0.
2. `smoke/fd1_r0.py` → F1; `smoke/fd2_mms.py` → F2; F3 trong mọi lượt.
3. Đo tốc độ thật và bộ nhớ ở 2.5 µm (một lượt ngắn) và ghi vào đây trước khi chạy đủ. Ước tính hiện tại (suy từ phép đo thô; chưa đo với nguồn, bay hơi, mặt nạ): 35–70 phút mỗi lượt 2.5 µm. Lượt 2.5 µm **chắc chắn có**: F1 (R0), F2 (MMS, có thể rút ngắn thời gian mô phỏng) và C1 ở giai đoạn B, khoảng 1.5–3 giờ GPU. Thêm C2–C4 ở 2.5 µm (giai đoạn C′) nếu quy tắc 4.2 đòi: tổng đến khoảng 4–8 giờ.
4. Giai đoạn A, B, C (hoặc C′) theo mục 4.2; ghi lưới tham chiếu đã chọn và e_FD **trước** khi chạy bước 5.
5. `experiments/fd_compare.py`: nạp checkpoint PINN, đánh giá trên lưới FD tham chiếu, ghi `runs/FD/summary.json` và bản đồ sai số. Phán quyết theo mục 4.3.
6. Cập nhật `TIEN_DO.md` (mục 3, 5, 6) và ghi kết quả + chấm dự đoán vào cuối file này.

Windows: chuỗi in trong mã dùng không dấu; lệnh dài ghi file rồi chạy; dùng `lock.py`; `gc.collect()` và `torch.cuda.empty_cache()` trước mỗi lượt GPU.

## 8. Giới hạn của 07 (khóa)

Chỉ nói PINN có giải đúng **PDE hiện tại** trong ngân sách 10% hiệu ứng, một điều kiện (300 W, 0.8 m/s), một hình học (45°, y_a = 0), mặt nạ cố định lấy từ PINN (phép lặp mặt nạ chưa kiểm). Ngân sách 10% và hệ số 3 là quy ước, không suy ra từ vật lý hay từ yêu cầu của hội đồng. Không nói gì về vật lý, thực nghiệm, độ nhạy theo giả định (σ ẩn nhiệt, k_bột, độ sâu miền, W nhỏ, ρCp sau chảy), hay mô hình lịch sử bột khác (melt, cons): các việc đó sau 07, cần đăng ký riêng.
