# PINN nhiệt LPBF bulk — giai đoạn 1

Mô hình dẫn nhiệt 3D phi ổn định cho một track laser Gaussian trên Ti-6Al-4V bulk,
không có latent heat, dùng hằng số vật liệu tại nhiệt độ tham chiếu. Mục tiêu là
kiểm chứng PDE, nguồn nhiệt và điều kiện biên trước khi thêm latent heat và k(T), c_p(T).

Mã được viết mới, không import từ các notebook hay module cũ trong repo.

## Quyết định đã chốt

| Mục | Giá trị | Nguồn / ghi chú |
|---|---|---|
| Miền | 800 × 400 × 290 µm³, bulk kim loại | Bột 30 µm coi là kim loại đặc (bỏ độ rỗng) |
| Đáy | z = 0, Dirichlet T = T₀ = 308 K | Khớp bài báo |
| Mặt phẳng substrate | z = 260 µm | Tham chiếu đo độ sâu (bài báo) |
| Mặt trên | z = 290 µm | Đối lưu + bức xạ + nguồn laser |
| Mặt bên (4 mặt) | Đối lưu + bức xạ | Đã chốt |
| Track | x: 100 → 700 µm, y = 200 µm, v = 750 mm/s | t_end = 0.8 ms |
| Laser | P = 150 W, r = 50 µm (exp(−2r²/R²)) | Gaussian bề mặt |
| Hấp thụ α | 0.41 | Fresnel với ε = 0.27 tại góc tới nhỏ (Ma et al. Fig. 2) |
| Bức xạ | α_r = 0.4 | Ma et al. Table 1 |
| Đối lưu | h = 25 W/(m²K) | Ma et al. Table 1 |
| Vật liệu | ρ = 4420, c_p = 558.5, k = 9.84 | Hằng số tại T_ref = 308 K, từ công thức Table 1 |
| Ẩn nhiệt | Không dùng (giai đoạn 1) | |
| Kiểm tra | T_max ≤ T_V = 3315 K | Báo cáo khi vượt |
| Melt pool | Đẳng nhiệt T_s = 1878 K | |
| Chuẩn so sánh | Bảng 4 (mô phỏng): W = 114.8 µm, D = 68.7 µm | Chuẩn chính |
| Tham khảo | Bảng 3 (thí nghiệm): W = 123.21 µm, D = 74.72 µm | |

## Phương trình

Dạng không thứ nguyên với x* = x/r, t* = t/t_c, t_c = r²/α, θ = (T − T₀)/T_c, T_c = 2000 K:

- PDE trong miền: ∂θ/∂t* = ∇*²θ
- Mặt trên (n = +z): ∂θ/∂z* = L(θ) − Q*·exp[−2((x*−x*_c)² + (y*−y*_c)²)]
- Mặt bên (n ngoài): ∂θ/∂n* = L(θ)
- Đáy: θ = 0
- Điều kiện đầu: θ = 0 tại t* = 0 (ép cứng)

Với L(θ) = Bi·θ + C_rad·[(T₀ + T_c·θ)⁴ − T₀⁴], Bi = hr/k, C_rad = εσr/(kT_c), Q* = 2αPr/(πr²kT_c).

Ghi chú về thứ nguyên: DeepXDE RobinBC có dạng dθ/dn = func(x, θ). Trong code, `physics.py` trả về đúng vế phải của biên.

## Điều kiện đầu (hard-IC)

θ = [1 − exp(−t*/τ*)] · softplus(f(x*, t*)), với τ = 0.05 ms (τ* ≈ 0.08).
- Mask đảm bảo θ = 0 tại t* = 0.
- softplus đảm bảo θ ≥ 0, tức T ≥ T₀. Đây là giả định hợp lý vì nguồn nhiệt dương và không có nguồn lạnh.

## Mạng và huấn luyện

- FNN 4 → 5×32 (tanh) → 1, Glorot uniform. Input chuẩn hóa về [−1, 1] theo miền.
- Điểm nội: Sobol 4D, z kéo lên gần mặt trên bằng z = Z_top·(1 − (1 − u)²).
- Điểm bám laser: Sobol 4D, mỗi điểm có thời điểm t, đĩa bán kính 3r quanh (x_c(t), y_c), độ sâu 3r dưới mặt trên.
- Điểm biên: DeepXDE lấy đều trên mặt biên.
- Loss: [PDE, mặt trên, mặt bên, đáy] với trọng số [1, 1/Q*², 1, 1]. Mặt trên có flux đỉnh Q* ≈ 40, nên trọng số bình phương giúp các số hạng cân bằng. Đây là lựa chọn của mình, có thể đổi trong `model.py`.
- Adam, lr = 1e-3, giảm theo thời gian nghịch đảo (`inverse time`, 2000 bước, tỉ lệ 0.5). Không dùng scheduler theo plateau.
- L-BFGS sau Adam.

Cấu hình:
- `smoke`: 2000 Adam + 300 L-BFGS, ít điểm, để kiểm tra code.
- `full`: 30000 Adam + 20000 L-BFGS, đầy đủ điểm.

## Cách chạy

```bash
pip install -r requirements.txt
python config.py                              # in tham số không thứ nguyên
python train.py --preset smoke --tag smoke_01 # huấn luyện ngắn
python evaluate.py --tag smoke_01 --t 0.4e-3 0.8e-3
```

Kết quả nằm trong `outputs/<tag>/`: `model.pt`, `meta.json`, `loss.npz`, `evaluation.json`.

## Hạn chế và việc để sau

- Không có bột, khí và latent heat. Hằng số vật liệu lấy tại 308 K cho cả pha lỏng, nên melt pool có thể lệch so với thí nghiệm, đặc biệt vì k_l = 33 không được dùng.
- Bảng 4 là mô phỏng CFD có Marangoni, recoil, latent heat và bột, nên chênh lệch với mô hình này phản ánh mức đơn giản hóa, không phải lỗi PINN.
- Track 0.6 mm chưa đạt trạng thái ổn định. Melt pool đo tại t_end khác với trạng thái dừng của bảng 3 và 4.
- Giai đoạn 2: thêm latent heat (enthalpy, làm trơn f_l), k(T), c_p(T), và k_l.
