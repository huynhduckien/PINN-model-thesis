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

---

## Sửa đổi A1 (2026-10-07 18:44, đồng hồ): F0 trượt lần đầu vì đặc tả sai đầu vào của ca tham chiếu; ngưỡng không đổi

**Kết quả lần chạy đầu của F0** (`smoke/fd0_regress.py`, `runs/FD/f0.json`, giữ làm bản ghi): ca `F5_old_h10` **đạt** (đỉnh trong 4e-16 tương đối; rộng/sâu/dài trong 5e-13 µm; đóng năng lượng 8e-14; 5250 bước, trùng cả số bước). Ca `OH0_a1s_overhang_h10` **trượt**: GPU chạy 5250 bước còn tham chiếu 5251; đỉnh lệch 8.6e-5 tương đối, rộng 0.0096 µm, sâu 0.0022 µm, dài 0.0285 µm (vượt 1e-9 và 1e-6 µm của tiêu chí khóa). Đóng năng lượng 8e-14.

**Chẩn đoán (đã kiểm bằng số):** số bước = ceil(t_end/dt₀) với t_end/dt₀ **bằng đúng số nguyên** 5250 về mặt toán học, nên ceil lật theo sai số làm tròn 1 ulp của h. Lượt tham chiếu `OH0` khởi chạy qua dòng lệnh `--h 10`, tức h = 10.0×1e-6 = 9.999999999999999e-06 (nhỏ hơn 1e-05 một ulp) → t_end/dt₀ = 5250.000000000001 → 5251 bước (khớp `report.json` của tham chiếu: `steps` 5251, `cfl_dt0_us` 0.2857142857142857). `F5_old_h10` dùng hằng 10e-6 → 5249.999999999998 → 5250 bước. Tôi dùng 10e-6 cho cả hai nên lệch một bước ở ca `OH0`. **Không phải lỗi bản chuyển sang GPU:** ca `F5_old_h10` trùng đến sai số làm tròn máy. Đây là lỗi đặc tả của tôi (không nhận ra t_end/dt₀ là số nguyên chính xác).

**Sửa (không đổi ngưỡng nào):**
1. F0: mỗi ca tham chiếu chạy với **đúng đầu vào h mà tham chiếu đã dùng**: `F5_old_h10` với h = 10e-6; `OH0_a1s_overhang_h10` với h = 10.0*1e-6. Ngưỡng giữ nguyên: đỉnh trong 1e-9 tương đối; rộng/sâu/dài trong 1e-6 µm; đóng năng lượng < 1e-9; **thêm** kiểm số bước bằng tham chiếu.
2. Cho các lượt chính (numerics `v2`) tránh đúng cái bẫy này: số bước = ceil(t_end/dt₀ − 1e-6) (xác định, không phụ thuộc 1 ulp của h). dt không vượt dt₀ quá 1e-6 tương đối, chắc chắn nằm trong hệ số an toàn 0.4.

Hai lượt F0 lần đầu giữ nguyên trong `runs/FD/f0.json` (đổi tên thành `f0_run1.json` trước khi chạy lại).


---

## Sửa đổi A2 (2026-10-07 18:54, đồng hồ): làm rõ đặc tả F1, F2 và cách tổ chức mã `v2`; **không đổi ngưỡng nào**

Viết **trước khi chạy F1 và F2**, sau khi viết mã nhưng chưa chạy mốc nào của F1, F2.

**1. Tổ chức mã.** Numerics `v2` nằm trong `fdv2.py` (không nằm trong `fdref.py`); `fdref.py` giữ nguyên numerics `old` cho phép hồi quy F0. Nguồn laser và κ_s được tính **dạng tách biến** (nguồn = tích hàm của (x, y) và hàm của z; κ_s = 1 + (r−1)·s_κ(y,z)·(1 − σ_x(x,t)·e(y,z))) vì gọi `q_volume` / `maskphys.kappa_static` trên lưới 3D ở mỗi bước quá chậm. Công thức là của hai hàm gốc; để không chép bản riêng mà chưa kiểm, `smoke/fd_coef.py` so với chính hai hàm đó tại mọi tâm ô lưới 20 µm ở ba thời điểm (30, 60, 90% quét), ba bậc (B3W0s, B3s, B3W0), tiêu chí sai lệch tương đối lớn nhất < 1e-12. **Kết quả đã chạy lúc ghi mục này: ĐẠT** (κ_s: 0; nguồn: 4.4e-16 và 5.1e-16; κ_s mặt trên: 0; `runs/FD/coef.json`). Dạng nguồn trong `solve` dùng tích phân chính xác theo z trên từng ô (`zmode = "exact"`, prereg 3.2); phép kiểm so điểm giữa (`zmode = "mid"`).

**2. F1 (làm rõ).** Chỉ báo của cả FD lẫn R0 đo bằng **cùng một hàm** `indicator.pool_dims`: FD qua nội suy tuyến tính theo tâm ô của trường lưới (kẹp ở biên), R0 qua nghiệm giải tích; tại **thời điểm của ảnh chụp 70% của chính lượt FD** (R0 tính lại ở thời điểm đó). Bước R0: `legacy/overhang_r0.py`, cấu hình `PHYS_A0` (bột tắt, ẩn nhiệt tắt, không bay hơi; α = 0.35, β = 1/40 µm). Tiêu chí giữ như bảng 4.4: ở 2.5 µm, |sâu_FD − sâu_R0| và |rộng_FD − rộng_R0| ≤ **0.61 µm**; sai số (sâu và rộng) giảm đơn điệu nghiêm ngặt khi lưới 10 → 5 → 2.5 µm; đóng năng lượng < 1e-9 ở cả ba lưới.

**3. F2 (làm rõ).** MMS kiểm bốn toán tử của PINN: `B3` (vệt thường, `L.pde`), `B3W0` (nêm bột như đặc, `L.pde` với mặt nghiêng), `B3s` và `B3W0s` (cách nhiệt: `maskphys.pde_masked` sau `install`, bao T_max của `runs/ENV/B3_iter0.pt` và `runs/Fs/B3W0s_it0/envelope_next.pt`, độ rộng chuyển tiếp 30 µm như `config.RUNS[...]["soft"]`). Để số hạng nguồn bù tính được ở mọi bước, dùng **hộp thu nhỏ** L_x = 200 µm, L_y = 240 µm, L_z = 240 µm (z và các hàm hệ số tính từ cùng vị trí mặt phân giới và bao như bản đầy đủ) và **cửa sổ thời gian 40 µs**; với hai bậc cách nhiệt đặt laser **đứng yên** (`Pe = 0`, mũi bể x_front = 0 = tâm hộp) để κ_s không đổi theo thời gian (mặt nạ vẫn nhân vào toán tử; chuyển động của mũi bể không có đạo hàm theo thời gian trong PDE). Nghiệm chế tạo (đơn vị SI, tọa độ x, y từ tâm hộp, z từ đáy, τ = t/40 µs):
T_ms = T_amb + ΔT·G_x·G_y·G_z·w(τ), ΔT = 4500 K, G_x = ½(1+cos(2πx/L_x)), G_y = ½(1+cos(2πy/L_y)), G_z = sin²(πz/(2L_z)), w = sin²(πτ/2).
Nghiệm bằng T_amb trên đáy và bốn mặt bên và tại t = 0, đạo hàm theo z bằng 0 ở mặt trên. Số hạng nguồn bù S = `L.pde`(θ_ms) + `L.q_volume`, tính bằng autograd ở mỗi bước tại tâm ô (nhân k T_c/r_b² để về W/m³); biên mặt trên: thông lượng ra = q_ev(T_s) + g, với g = −q_ev(T_ms mặt trên) để T_ms thỏa. Sai số: RMS của (T_FD − T_ms) trên mọi ô tại t = 40 µs. **Bậc quan sát p = log₂(e₅/e₂.₅) phải nằm trong [1.7, 2.3] ở cả bốn toán tử** (khoảng 2 ± 0.3 của bảng 4.4); ghi thêm p giữa 10 và 5 µm. **Mốc bậc cho 4.2:** p_MMS của ca C1, C2, C3, C4 là p quan sát (5 → 2.5 µm) của toán tử cùng tên (`B3`, `B3W0`, `B3s`, `B3W0s`).

**4.** Các mục khác của 07 giữ nguyên.


---

## Sửa đổi A3 (2026-10-07 19:01, đồng hồ): F1 đo ở lưới mịn nhất đã chạy; quy tắc lưới thích ứng cho F2 (viết trước khi chạy F2)

**Bối cảnh và công khai về thời điểm sửa.** Người dùng quyết: "nếu chạy tốt thì không cần 2.5 µm". F1 đã chạy ở 10 và 5 µm **trước** mục này và cho (`runs/FD/f1.json`): ở 10 µm: sâu_FD − sâu_R0 = −0.213 µm, rộng −0.259 µm, đóng năng lượng 3.9e-15; ở 5 µm: sâu −0.020 µm, rộng −0.087 µm, đóng năng lượng 2.1e-14 (R0 trùng S0.4: sâu 85.94, rộng 122.00, dài 520.16 µm). Tức đoạn sửa dưới đây **được viết sau khi đã thấy kết quả 10 và 5 µm của F1**, theo quyết định của người dùng, không phải đăng ký trước thuần túy. Không nới ngưỡng dung sai nào: nó chỉ đổi lưới dùng để chấm.

**1. F1 (sửa).** Tiêu chí chấm ở **lưới mịn nhất đã chạy** thay vì cố định 2.5 µm: |sâu_FD − sâu_R0| ≤ **0.61 µm** và |rộng_FD − rộng_R0| ≤ **0.61 µm** ở lưới đó; sai số (sâu và rộng) giảm nghiêm ngặt qua các lưới đã chạy; đóng năng lượng < 1e-9 ở mọi lưới đã chạy. Lưới 2.5 µm **không** được chạy cho F1; vì vậy **không xác nhận** tính đơn điệu qua ba lưới và không có đo bậc hội tụ trên R0 (chỉ hai điểm). Chấm ở lưới thô hơn không làm tiêu chí dễ hơn (sai số ở 5 µm lớn hơn ở 2.5 µm).

**2. F2 (quy tắc lưới thích ứng, khóa trước khi chạy).** Chạy từng toán tử ở 10 và 5 µm. Tính bậc quan sát p₁₀→₅ = log₂(e₁₀/e₅) (RMS). **Nếu p₁₀→₅ nằm trong [1.7, 2.3] ở cả bốn toán tử: F2 ĐẠT**, không chạy 2.5 µm (dùng p₁₀→₅ làm p_MMS cho mục 4.2). **Nếu bất kỳ toán tử nào ngoài khoảng đó:** chạy 2.5 µm cho **toán tử đó** (hộp thu nhỏ, rẻ), chấm theo p₅→₂.₅ ∈ [1.7, 2.3] đúng như bảng 4.4; nếu vẫn ngoài khoảng thì F2 TRƯỢT cho toán tử đó. Quy tắc dừng này chỉ dựa trên kết quả MMS, không dựa trên PINN.

**3.** Quy tắc lưới thích ứng của mục 4.2 (giai đoạn B: C1 ở 2.5 µm luôn chạy) **chưa được sửa ở đây**; sẽ bàn riêng khi tới bước C1–C4, vì F1 hội tụ nhanh ở 5 µm không chứng minh được sơ đồ có bay hơi, ẩn nhiệt và mặt nạ cũng hội tụ nhanh.


---

## Sửa đổi A4 (2026-10-07 19:15, đồng hồ): F0, F1, F2 và `fd_coef` là chẩn đoán, không phải cổng chặn; bỏ khoảng bậc của F2

**Nguyên tắc (người dùng):** FD chỉ để xác minh PINN có giải đúng PDE. Kết luận đúng/sai **chỉ** dựa vào so sánh PINN–FD trong ngân sách của mục 4.3, đã có tính sai số FD (e_FD). Nếu PINN nằm trong sai số cho phép thì PINN vượt qua phép kiểm.

**Hệ quả (thay các điều kiện "một cái trượt thì dừng" ở mục 4.4, và mục 4.4 F2 / A3 điểm 2):**
1. F0 (hồi quy FD cũ), F1 (R0), F2 (MMS) và `fd_coef` là **chẩn đoán**: dùng để **quy lỗi** khi PINN và FD lệch nhau (lỗi ở PINN hay ở FD). Chúng không phải điều kiện để được chạy bước 5. Lý do: nếu FD giải một PDE khác `L.pde` thì kết quả là PINN lệch FD (sai quy cho PINN); nếu PINN và FD khớp thì phán quyết ĐÚNG đã có hai bộ giải độc lập đồng ý, và các chẩn đoán chỉ bổ sung. Nếu một chẩn đoán cho thấy không hội tụ (sai số không giảm theo lưới) thì phải điều tra **trước khi diễn giải** một phán quyết SAI hoặc KHÔNG KẾT LUẬN.
2. **Bỏ khoảng bậc [1.7, 2.3] của F2** làm tiêu chí; bậc quan sát chỉ được ghi như số liệu. Không chạy MMS ở 2.5 µm.
3. Ước sai số FD dùng p_dùng = min(p_quan sát, 2) trong công thức GCI của mục 4.2 (nghiêm hơn, không nới: e_FD lớn hơn).
4. Quy tắc chống thiên lệch của 4.2 giữ nguyên: lưới tham chiếu chỉ chọn theo hội tụ của chính FD, không theo việc PINN và FD có khớp hay không.

**Công khai về thời điểm:** mục này viết **sau** khi đã có kết quả chẩn đoán nhưng **trước** khi có bất kỳ số so sánh PINN–FD nào. Phán quyết chính (mục 4.3) chưa được động tới nên các thay đổi trên không thể thiên lệch nó. Kết quả chẩn đoán đã có: F0 trùng FD cũ tới 4e-16; F1 (A3): sâu −0.020 µm, rộng −0.087 µm ở 5 µm (so ngưỡng tham khảo 0.61 µm), đóng năng lượng 2e-14; `fd_coef`: κ_s lệch 0, nguồn 5e-16; F2 (MMS, 10 → 5 µm): RMS 25.0 → 4.76 K (`B3`, p 2.39), 25.8 → 4.89 K (`B3W0`, p 2.40), 19.1 → 4.14 K (`B3s`, p 2.21), 19.7 → 3.96 K (`B3W0s`, p 2.31).


---

## Sửa đổi A5 (2026-10-07 21:35, đồng hồ): bỏ giai đoạn B, lưới tham chiếu 5 µm, phán quyết hai cận, Δ không xác định

**Công khai về thời điểm.** Mục này viết **sau** khi đã thấy: kết quả FD giai đoạn A (10 và 5 µm, `runs/FD/fd_cases.json`), chu trình nhiệt của PINN tính lại từ checkpoint (`runs/FD/pinn_cycles.json`) và phép chẩn đoán cục bộ (`runs/FD/fieldcheck.json`), nên tôi đã biết ở mức sơ bộ PINN và FD lệch nhau thế nào (ΔD cách nhiệt PINN +18.31 µm so với FD khoảng +26.8 µm). **Phán quyết chính thức (`fd_compare.py compare`) chưa chạy** vào lúc ghi mục này. Quyết định bỏ giai đoạn B là của người dùng ("khóa phán quyết", chọn phương án (b) trong hai phương án tôi nêu); lý do tôi nêu: giai đoạn B tốn ước 2–3 giờ GPU và ΔD cách nhiệt lệch ~8.5 µm so với sai số FD ước chỉ ≤ 0.8 µm ngay cả khi giả định bậc hội tụ xấu nhất (p = 1).

**1. Giai đoạn B bỏ; lưới tham chiếu = 5 µm** (lưới mịn nhất đã chạy cho cả bốn ca). Không có đo bậc hội tụ trên bài thật (bộ ba 20, 10, 5 µm của C1 chỉ cho kết quả không kết luận được: độ sâu không đơn điệu, độ rộng bậc ~1).

**2. Phán quyết hai cận** (thay phép tính e_FD của mục 4.2/4.3 và A4 điểm 3): với mỗi Δ_Q tính sai số FD hai lần, e(p) = 1.25·|Q₅ − Q₁₀|/(2^p − 1) với **p = 2** (A4) và **p = 1** (xấu nhất); chấm trạng thái ĐÚNG/SAI/KHÔNG KẾT LUẬN theo mục 4.3 ở từng cận. Trạng thái của Δ_Q là trạng thái chung nếu hai cận **cùng** kết luận; nếu khác nhau thì KHÔNG KẾT LUẬN. Ngân sách tol_Q (10% hiệu ứng của PINN) và mục tiêu tol_Q/3 giữ nguyên; mục tiêu tol_Q/3 chỉ còn là con số báo cáo, không còn quyết định lưới.

**3. Δ không xác định:** Δ_Q có giá trị không xác định ở FD (NaN ở một trong hai lưới; xảy ra với tỉ số tốc độ nguội của nêm cách nhiệt, vì điểm chu trình vẫn trên T_liq đến hết cửa sổ mô phỏng 1.5 ms: FD cho T = 1640 K tại t_end, PINN 1406 K) ghi **KHÔNG XÁC ĐỊNH**, không tham gia quyết định lưới, và **không** biến bảng thành ĐÚNG.

**4. Phán quyết bảng chỉ báo (C4 − C3, bốn Δ chính):** ĐÚNG chỉ khi cả bốn Δ ĐÚNG; SAI nếu có ít nhất một Δ SAI; còn lại KHÔNG KẾT LUẬN. Δ phụ (C2 − C1, ΔD bột như đặc) chấm riêng. Các số của phép chẩn đoán cục bộ (trường nhiệt, chu trình nhiệt, phần dư PDE của PINN, năng lượng) là **bằng chứng bổ sung**, không tham gia phán quyết.

**5.** Hành động sau phán quyết giữ nguyên mục 4.3: không tự huấn luyện lại; nếu SAI hoặc KHÔNG KẾT LUẬN thì bảng chỉ báo ghi cả hai cột (PINN, FD) cho đến khi người dùng quyết.


---

## Kết quả 07 (2026-10-07 21:36, đồng hồ): phán quyết SAI cho bảng chỉ báo cách nhiệt (C4 − C3)

Chạy theo A5: `fd_compare.py grid` (quyết định lưới chỉ dựa trên FD: lưới tham chiếu 5 µm), rồi `fd_compare.py compare` (`runs/FD/grid_decision.json`, `runs/FD/compare.json`). Sai số FD ước ở hai cận p = 2 và p = 1; ngân sách tol = 10% hiệu ứng của PINN.

| Δ (nêm − vệt thường) | PINN | FD (5 µm) | \|lệch\| | e_FD (p=2 / p=1) | tol | p = 2 | p = 1 | Phán quyết |
|---|---|---|---|---|---|---|---|---|
| ΔD cách nhiệt (µm) | +18.31 | **+26.77** | 8.46 | 0.27 / 0.80 | 1.83 | SAI | SAI | **SAI** |
| ΔT_pk (K) | +159.8 | +179.9 | 20.1 | 0.93 / 2.79 | 15.98 | SAI | SAI | **SAI** |
| Δ thời gian lưu (µs) | +222.0 | +230.3 | 8.3 | 1.15 / 3.44 | 22.2 | ĐÚNG | ĐÚNG | **ĐÚNG** |
| tỉ số tốc độ nguội | 0.5425 | không xác định | – | – | 0.0457 | – | – | **KHÔNG XÁC ĐỊNH** |
| ΔD bột như đặc (µm) | +8.99 | +9.62 | 0.63 | 0.11 / 0.34 | 0.90 | ĐÚNG | KHÔNG KẾT LUẬN | **KHÔNG KẾT LUẬN** |

**Phán quyết bảng chỉ báo (C4 − C3): SAI** (hai Δ SAI ở cả hai cận). **ΔD bột như đặc (C2 − C1): KHÔNG KẾT LUẬN.** Mục tiêu độ chính xác FD tol/3 đạt ở cả ΔD cách nhiệt (p = 2: 0.27 ≤ 0.61), ΔT_pk, Δ lưu; không đạt ở ΔD cách nhiệt khi p = 1 (0.80 > 0.61) và ở ΔD bột như đặc khi p = 1 (0.34 > 0.30).

**Tỉ số nguội không xác định ở FD:** điểm chu trình của C4 vẫn trên T_liq đến hết cửa sổ 1.5 ms (FD: 1640 K tại t_end; PINN: 1406 K); đây là cửa sổ cắt cụt đã ghi trong `TIEN_DO.md`, nay thấy ở cả PINN lẫn FD ở mức khác nhau.

**Bằng chứng bổ sung (chẩn đoán, không tham gia phán quyết)** — `runs/FD/fieldcheck.json`, `pinn_cycles.json`, so FD ở 10 µm:
- Nhiệt độ: PINN lạnh hơn FD ở mọi ca; trung bình (T_PINN − T_FD) trên vùng T > 1000 K tại 70% quét: C1 −11…−17 K, C2 −26…−54 K, C3 −96…−139 K, C4 −122…−197 K; T tại t = 1500 µs ở điểm chu trình 50%: PINN/FD = 1151/1204 (C1), 1202/1268 (C2), 1186/1350 (C3), 1406/1640 K (C4).
- Năng lượng lưu trữ (cùng định nghĩa): PINN thấp hơn FD 12.3 mJ (C2), 16.3 mJ (C3), 9.5 mJ (C4), tức 6–12%.
- Phần dư PDE của chính PINN (chuẩn hóa theo |C(1+latent)θ_t| + |q|, ~350 điểm gần chùm): trung vị 5% (C1), 12% (C2), 17% (C3), 19% (C4); phân vị 90 là 12%, 28%, 64%, 77%.
- Chẩn đoán F0 (trùng FD cũ tới 4e-16), F1 (R0: sâu −0.020 µm, rộng −0.087 µm ở 5 µm), `fd_coef` (κ_s lệch 0, nguồn 5e-16), F2 (MMS 10 → 5 µm: bậc 2.21–2.40) đều không cho thấy vấn đề ở FD. Bộ ba 20, 10, 5 µm của C1 không kết luận được bậc trên bài thật (độ sâu không đơn điệu ±0.4 µm, độ rộng bậc ~1).

**Chấm dự đoán (mục 6):**

| Dự đoán | Kết quả |
|---|---|
| F0–F3 đều đạt (~70%) | đạt về chẩn đoán, nhưng F1 phải sửa đặc tả (A3) và F2 bỏ khoảng bậc (A4) |
| dừng ở 5 µm không cần C2–C4 ở 2.5 µm (~20%) | dừng ở 5 µm, nhưng nhờ A5 (bỏ giai đoạn B), không phải nhờ quy tắc hội tụ |
| đạt mục tiêu tol/3: ΔD ~50%, lưu ~40%, tỉ số nguội ~35%, ΔT_pk ~25% | ΔD đạt ở p = 2 (không đạt ở p = 1); lưu đạt; ΔT_pk **đạt** (dự đoán sai: 0.93 K ≪ 5.3 K); tỉ số nguội không xác định (dự đoán sai: không nghĩ tới cửa sổ cắt cụt) |
| phán quyết bảng: ĐÚNG ~30%, KHÔNG KẾT LUẬN ~35%, SAI ~35% | **SAI** (đúng loại có xác suất thấp nhất tôi ghi, 35%) |
| ΔD bột như đặc KHÔNG KẾT LUẬN ~50% | **KHÔNG KẾT LUẬN** (đúng) |
| FD hội tụ C1: sâu 88–92 µm, rộng 165–169 µm (~70%) | đúng: 90.57 và 167.71 µm |
| nếu có Δ SAI, ΔD của FD lớn hơn PINN (~60%) | đúng: +26.77 so với +18.31 µm |

**Giới hạn của kết quả này:** một điều kiện (300 W, 0.8 m/s), góc 45°, y_a = 0; mặt nạ cố định lấy từ PINN (phép lặp mặt nạ chưa kiểm); hai lưới (10 và 5 µm), không đo bậc hội tụ trên bài thật (giai đoạn B bỏ theo A5); tỉ số tốc độ nguội không đánh giá được; MMS dùng mặt nạ đứng yên nên chuyển động mũi bể chỉ được kiểm bằng so khớp hàm hệ số (`fd_coef`). Phán quyết chỉ nói về việc PINN có giải đúng **PDE hiện tại** hay không, không nói PDE có đúng vật lý hay không. Phán quyết SAI không phụ thuộc vào bậc hội tụ giả định (p = 2 và p = 1 cùng kết luận) nhưng phụ thuộc vào mức sai số FD không vượt xa hai cận đó.
