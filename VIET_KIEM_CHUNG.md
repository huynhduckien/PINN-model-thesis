# Kiểm chứng số của mô hình PINN (bản nháp để ghép vào luận văn)

Soạn 2026-10-09, dựa trên `prereg/07`–`13`, `runs/FD/`, `runs/Figures/pinn_vs_fd/`. Đặt như một mục con trong chương Phương pháp (ví dụ "3.x Kiểm chứng số"), không phải chương riêng. Số liệu lấy trực tiếp từ file kết quả, không điền theo trí nhớ.

---

## 1. Vì sao cần kiểm chứng độc lập, không chỉ dựa vào loss

Phần dư của phương trình đạo hàm riêng (PDE residual) trong hàm mất mát của PINN thường được xem là thước đo mức độ mạng thỏa phương trình. Tuy nhiên, với bài toán nhiệt LPBF có nguồn di chuyển và gradient dốc, chúng tôi đo được trực tiếp rằng loss PDE thấp **không đồng nghĩa** với nghiệm chính xác:

- Một mạng được khớp có giám sát để tái tạo đúng trường nhiệt của bộ giải tham chiếu (sai số RMS chỉ 11–13 K) có loss PDE là **4.0–4.2**.
- PINN huấn luyện bằng cách giảm phần dư PDE (không dùng dữ liệu) đạt loss PDE rất thấp, **0.0055–0.0087**, nhưng sai số so với bộ giải tham chiếu là **135–226 K**.
- Xuất phát từ mạng chính xác ở trên và tiếp tục giảm loss PDE bằng hạ gradient, sai số **tăng** lên 460–570 K trước khi có dấu hiệu giảm trở lại.

Ba quan sát này cho thấy quan hệ giữa loss PDE và độ chính xác không đơn điệu trong bài toán này: tồn tại các vùng trọng số có loss PDE thấp nhưng nghiệm sai lệch nhiều. Vì vậy, việc tuyên bố "PINN giải đúng PDE" cần một phép đo độc lập với loss huấn luyện.

## 2. Bộ giải tham chiếu và sáu bước kiểm chứng

Một bộ giải sai phân hữu hạn (finite-volume, GPU) được xây dựng độc lập, dùng **đúng các hàm hệ số, nguồn và điều kiện biên** đã định nghĩa cho PINN (hệ số dẫn nhiệt κ_s(T, y, z), nguồn Beer–Lambert, biên bay hơi Anisimov/Fang). Việc dùng chung hàm vật lý đảm bảo khi so sánh hai bên, chênh lệch chỉ có thể đến từ sai số giải số của một trong hai bên, không phải từ việc so sánh hai mô hình vật lý khác nhau.

Bộ giải được kiểm chứng qua sáu bước, mỗi bước loại trừ một loại lỗi khác nhau:

| Bước | Phép kiểm | Kết quả | Loại lỗi bị loại trừ |
|---|---|---|---|
| 1 | Hồi quy so với bộ giải sai phân đã dùng trước đây | lệch 4×10⁻¹⁶ | lỗi khi chuyển mã sang GPU |
| 2 | So với nghiệm giải tích (phương pháp ảnh, bài toán rút gọn) | độ sâu lệch −0.02 µm, độ rộng lệch −0.09 µm (lưới 5 µm) | lỗi rời rạc hóa cơ bản |
| 3 | Dùng chung hàm hệ số, nguồn, κ_s với PINN | lệch 0 đến 5×10⁻¹⁶ | so sánh hai mô hình vật lý khác nhau |
| 4 | Phương pháp nghiệm nhân tạo (MMS) với **chính toán tử `L.pde` của PINN** | bậc hội tụ 2.2–2.4 | lỗi rời rạc hóa của đúng phương trình PINN đang giải |
| 5 | Đóng năng lượng | sai số tương đối ~10⁻¹³ | thất thoát năng lượng trong mã |
| 6 | Hội tụ lưới (10 µm và 5 µm) | phán quyết không đổi dù giả định bậc hội tụ p = 2 hay p = 1 | sai số rời rạc hóa không chi phối kết luận |

**Phạm vi của phép kiểm chứng này.** Sáu bước trên xác nhận bộ giải tham chiếu **giải đúng phương trình đã định nghĩa**, không xác nhận phương trình đó mô tả đúng vật lý của LPBF thực tế (không có dòng chảy nóng chảy, không có khoang hơi do áp suất phản lực). Đây là **kiểm chứng (verification: có giải đúng phương trình đã viết ra không), không phải xác nhận (validation: phương trình đó có đúng thực tế không)**. Việc xác nhận với thực nghiệm nằm ngoài phạm vi nghiên cứu này do chưa có dữ liệu ngoài phù hợp với đúng điều kiện công nghệ khảo sát.

## 3. Quy tắc chọn PINN của hồ sơ

Để tránh việc chọn mạng có kết quả đẹp sau khi đã thấy số đối chiếu, PINN của hồ sơ cho mỗi hình học được chọn theo quy tắc đặt **trước** khi đối chiếu với bộ giải tham chiếu: **checkpoint được huấn luyện nhiều nhất mà đạt đủ cổng vật lý đặt trước** (cân bằng năng lượng |E_res| và |E_id| ≤ 10 mJ; nhiệt độ hụt dưới môi trường ≤ 1% biên độ trên 2²¹ điểm; tỉ lệ năng lượng tích lũy và bay hơi trên năng lượng nạp). Áp dụng quy tắc này: vết thường cách nhiệt dùng checkpoint huấn luyện dài (ký hiệu F08 trong các bảng dưới), nêm cách nhiệt dùng checkpoint huấn luyện với phạt dương nhắm vào điểm cực trị để đạt cổng (ký hiệu F13).

## 4. Đối chiếu PINN với bộ giải tham chiếu

Sai số tại 70% quãng đường quét, trường nhiệt trên vùng T > 600 K (`verif_table.json`); độ sâu/độ rộng/chiều dài bể chảy (vùng T ≥ T_solidus) lấy từ lưới 5 µm.

| Ca | Hình học | Bột | L2 vùng nóng | RMS vùng nóng (K) | Lệch TB vùng nóng (K) | Độ sâu (lệch) | Độ rộng (lệch) | Chiều dài (lệch) |
|---|---|---|---|---|---|---|---|---|
| C1 | vệt thường | như đặc | 4.4% | 42 | −20 | +0.2% | −0.3% | −5% |
| C2 | nêm | như đặc | 11.0% | 102 | −72 | −0.4% | −0.6% | −13% |
| C3 | vệt thường | cách nhiệt (F08) | 12.7% | 135 | −100 | −0.5% | −4.0% | −24% |
| C4 | nêm | cách nhiệt (F13) | 18.9% | 226 | −149 | −0.7% | −5.7% | −26% |

Hiệu ứng nêm theo độ sâu (ΔD, nêm − vệt thường, 70% quét): PINN +26.4 µm, bộ giải tham chiếu +26.8 µm. Bột như đặc: PINN +9.0 µm so với +9.6 µm.

**Đọc bảng:** PINN tái tạo tốt vùng quanh chùm và đến khoảng 250 µm phía sau chùm ở cả bốn trường hợp; độ sâu lệch dưới 1% trong mọi trường hợp. Sai số tập trung ở phần đuôi bể chảy (khoảng cách xa chùm theo cả không gian lẫn thời gian): chiều dài bể chảy ngắn hơn 13–26%, và PINN lạnh hơn bộ giải tham chiếu một cách có hệ thống ở vùng nóng (lệch trung bình −20 đến −149 K). Mức lệch tăng dần khi bột chuyển từ dẫn nhiệt như đặc sang cách nhiệt, và khi hình học chuyển từ vệt thường sang nêm.

*(Chèn Hình 1–4 cho C3 và C4 tại đây: độ sâu/độ rộng theo khoảng cách sau chùm; mặt cắt ngang tại 73 µm; mặt cắt dọc qua trục chùm; đường nhiệt độ theo độ sâu và theo x — `runs/Figures/pinn_vs_fd/`.)*

## 5. Thảo luận: so với tài liệu PINN-LPBF đã công bố

Hiện tượng sai số tăng ở vùng xa nguồn nhiệt/xa dữ liệu huấn luyện không riêng của mô hình này. Sharma và cộng sự (2024, dẫn nhiệt–động lượng ghép Marangoni cho LPBF, r_b = 40 µm và v = 0.8 m/s — trùng điều kiện khảo sát ở đây) báo cáo cùng dạng suy giảm: dự đoán PINN khớp tốt gần mốc thời gian có dữ liệu huấn luyện (t = 280 µs) nhưng đỉnh nhiệt độ và kích thước bể chảy bị thu nhỏ, gradient dốc bị làm mịn khi ngoại suy xa hơn (t = 600 µs). Điểm khác biệt: mô hình của Sharma ghép dữ liệu CFD trực tiếp vào hàm mất mát (cả nhiệt độ lẫn vận tốc), và chuẩn so sánh của họ là chính phần mềm tạo ra dữ liệu đó, không độc lập. PINN trong nghiên cứu này không dùng dữ liệu nào trong huấn luyện, và được đối chiếu với một bộ giải độc lập đã qua sáu bước kiểm chứng ở mục 2 — do đó phép đối chiếu chặt hơn về phương pháp, dù phạm vi vật lý hẹp hơn (không có dòng chảy trong bể chảy).

## 6. Giới hạn (nêu rõ, không chờ bị phát hiện)

- **Kiểm chứng, không phải xác nhận.** Sáu bước ở mục 2 xác nhận bộ giải tham chiếu đúng với phương trình đã định nghĩa; không xác nhận phương trình đó đúng với LPBF thực tế. Mô hình không có dòng chảy trong bể chảy, không có khoang hơi do áp suất phản lực hơi; "bể chảy" là vùng T ≥ T_solidus trong một mô hình dẫn nhiệt, không phải bể chảy vật lý thật.
- **Hai lưới (10 µm, 5 µm).** Giai đoạn đo bậc hội tụ ở lưới mịn hơn trên bài toán thật đã được lược bỏ sau khi thấy số liệu sơ bộ. Phán quyết ở mục 2 không đổi dù dùng bậc hội tụ lý thuyết (p = 2) hay bậc thận trọng hơn (p = 1).
- **Mặt nạ vùng đã nóng chảy không độc lập hoàn toàn ở hai trường hợp cách nhiệt.** Mặt nạ dùng trong bộ giải tham chiếu lấy cố định từ bao nhiệt độ của chính PINN đã huấn luyện. Vì vậy phép so sánh ở các ca cách nhiệt kiểm tra việc PINN giải đúng PDE với mặt nạ này, chưa kiểm tra mặt nạ có đúng vật lý hay không.
- **Hệ số dẫn nhiệt hiệu dụng của bột chưa chắc chắn.** Giá trị dùng trong mô hình (bột dẫn nhiệt bằng khoảng 2.3% vật liệu đặc, từ tương quan Moser) khác đáng kể so với một tương quan độc lập khác trong tài liệu (khoảng 27.7%, dựa trên độ xốp — Sharma và cộng sự, 2025). Độ nhạy của kết quả theo hệ số này chưa được khảo sát.
- **Checkpoint ở nêm cách nhiệt (F13) đạt cổng vật lý nhờ một ngoại lệ đã công khai** (dừng sớm theo tiêu chí không đạt do lỗi thiết kế ngưỡng, được đánh giá lại như một checkpoint hoàn chỉnh — `prereg/13` mục A1–A2), và dùng một số hạng phạt dương sửa đổi không có đối chứng huấn luyện thêm cùng số bước mà không có sửa đổi đó.
- **Một seed cho mỗi cấu hình huấn luyện.**
