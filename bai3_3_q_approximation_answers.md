# Bài 3.3 - Q-function Approximation với Neural Network: Trả lời câu hỏi

---

## Câu 1: Vì sao Q-table không phù hợp với state liên tục?

Q-table lưu trữ giá trị Q(s, a) cho **mọi cặp (state, action)** trong một bảng tra cứu.

### Vấn đề với state liên tục:

1. **Vô hạn trạng thái**: State liên tục có vô hạn giá trị → bảng vô hạn dòng → không thể lưu trữ.
   - Ví dụ: CartPole state = [2.31, 0.47, -0.012, 0.89] — mỗi giá trị thực khác nhau đều là state khác.

2. **Discretization gây mất thông tin**:
   - Chia state thành bins → state gần nhau bị gộp chung → mất tính chính xác.
   - CartPole 10 bins × 4 chiều = 10,000 states — quá thô sơ.
   - Tăng bins: 100^4 = 100 triệu states — quá lớn, không thể thăm hết.

3. **Curse of dimensionality**: Số state tăng **hàm mũ** theo số chiều.
   - 10 chiều × 10 bins = 10^10 = 10 tỷ states.
   - Không đủ dữ liệu để cập nhật Q cho mỗi state.

4. **Không tổng quát hóa**: Q-table không chia sẻ thông tin giữa các state gần nhau.
   - State (2.31, 0.5) và (2.32, 0.5) hoàn toàn độc lập trong Q-table.
   - Phải thăm và cập nhật riêng từng state.

---

## Câu 2: Ý tưởng của function approximation là gì?

**Function approximation** thay thế bảng tra cứu bằng một **hàm xấp xỉ** có tham số:

```
Q-table:   Q(s, a) = tra bảng → giá trị
Q-approx:  Q(s, a) ≈ Q_θ(s, a) = f(s; θ) → giá trị
```

### Ý tưởng chính:
1. **Tham số hóa Q**: Dùng một hàm `f` có tham số `θ` (weights) để xấp xỉ Q(s, a).
2. **Tổng quát hóa**: Các state gần nhau sẽ có Q-value gần nhau — hàm `f` tự nội suy.
3. **Compact representation**: Thay vì lưu N×M giá trị (N states × M actions), chỉ cần lưu `θ` (vài nghìn parameters).
4. **Học = tối ưu θ**: Thay vì cập nhật từng ô trong bảng, ta cập nhật θ bằng gradient descent.

### Các dạng function approximation:
- **Linear**: Q(s, a) = w^T · φ(s, a) — đơn giản, giới hạn.
- **Neural Network**: Q(s, a) = NN_θ(s) — mạnh mẽ, phi tuyến.
- **Kernel methods, Gaussian processes**: Ít phổ biến trong RL.

---

## Câu 3: Sự khác biệt giữa Q-table và Q-network

| Tiêu chí | Q-table | Q-network |
|---|---|---|
| **Biểu diễn** | Bảng tra cứu (dictionary) | Mạng neural (weights) |
| **Bộ nhớ** | O(|S| × |A|) — tăng theo state space | O(|θ|) — cố định, không phụ thuộc |S| |
| **State liên tục** | Cần discretization | Xử lý trực tiếp |
| **Tổng quát hóa** | Không — mỗi state độc lập | Có — state gần nhau → Q gần nhau |
| **Tốc độ học** | Nhanh cho state space nhỏ | Chậm hơn (cần nhiều data + GPU) |
| **Scalability** | Kém — explodes với state lớn | Tốt — scale lên state space lớn |
| **Cập nhật** | `Q[s,a] += α · δ` (trực tiếp) | `θ -= lr · ∇L` (gradient descent) |
| **Stability** | Ổn định (convergence đảm bảo) | Kém ổn định hơn (có thể diverge) |
| **Cần replay buffer** | Không | Có (để ổn định training) |

---

## Câu 4: Giải thích kiến trúc mạng neural dùng để xấp xỉ Q(s, a)

```
Input Layer (4 neurons)     ← state vector [cart_pos, cart_vel, pole_angle, angle_vel]
         ↓
Hidden Layer 1 (128 neurons + ReLU)
         ↓
Hidden Layer 2 (128 neurons + ReLU)
         ↓
Output Layer (2 neurons)    ← Q-values [Q(s, left), Q(s, right)]
```

### Chi tiết:

| Layer | Kích thước | Vai trò |
|---|---|---|
| **Input** | 4 | Nhận state vector từ CartPole |
| **Hidden 1** | 128 + ReLU | Trích xuất features phi tuyến từ state |
| **Hidden 2** | 128 + ReLU | Học biểu diễn trừu tượng hơn |
| **Output** | 2 | Q-value cho mỗi action (left/right) |

### Code PyTorch:
```python
self.network = nn.Sequential(
    nn.Linear(4, 128),    # 4*128 + 128 = 640 params
    nn.ReLU(),
    nn.Linear(128, 128),  # 128*128 + 128 = 16,512 params
    nn.ReLU(),
    nn.Linear(128, 2),    # 128*2 + 2 = 258 params
)
# Tổng: ~17,410 parameters
```

---

## Câu 5: Vì sao output layer có số node bằng số action?

### Thiết kế output = |A|:
```
Q_network(s) → [Q(s, a_0), Q(s, a_1), ..., Q(s, a_{|A|-1})]
```

### Lý do:

1. **Hiệu quả tính toán**: Chỉ cần **1 lần forward pass** để có Q cho TẤT CẢ actions.
   - Thay vì input (s, a) → 1 giá trị Q (cần |A| lần forward pass).

2. **Dễ lấy max**: `max_a Q(s, a) = max(output)` — một phép toán đơn giản.
   ```python
   q_values = q_net(state)       # [Q(s,0), Q(s,1)]
   best_action = q_values.argmax()  # 0 hoặc 1
   ```

3. **Batch-friendly**: Có thể tính Q cho batch states cùng lúc:
   ```python
   # states: [batch, 4] → q_values: [batch, 2]
   q_values = q_net(states)
   ```

4. **Phổ biến trong DQN**: Kiến trúc này là chuẩn trong Deep Q-Network (Atari, 2015).

---

## Câu 6: Vai trò của hàm ReLU trong mạng

### ReLU (Rectified Linear Unit):
```
ReLU(x) = max(0, x) = { x  nếu x > 0
                       { 0  nếu x ≤ 0
```

### Vai trò:

1. **Tạo tính phi tuyến (non-linearity)**:
   - Nếu không có activation, nhiều linear layers = 1 linear layer → mạng chỉ học hàm tuyến tính.
   - ReLU cho phép mạng học **hàm phi tuyến phức tạp** (Q-function thường phi tuyến).

2. **Tránh vanishing gradient**:
   - Sigmoid/Tanh: gradient → 0 khi |x| lớn → học chậm.
   - ReLU: gradient = 1 khi x > 0 → gradient flow tốt.

3. **Sparse activation**:
   - ReLU "tắt" (= 0) cho x ≤ 0 → nhiều neurons không active.
   - Giúp mạng học biểu diễn hiệu quả.

4. **Tính toán nhanh**:
   - ReLU chỉ cần so sánh với 0 — nhanh hơn sigmoid/tanh (cần exp).

### Nhược điểm:
- **Dying ReLU**: Neuron bị "chết" nếu luôn nhận input ≤ 0 → gradient = 0 → không học.
- Giải pháp: Leaky ReLU, ELU, GELU.

---

## Câu 7: Giải thích hàm loss trong Q-learning với neural network

### Loss function:
```
L(θ) = E[(r + γ · max_a' Q_θ(s', a') - Q_θ(s, a))²]
```

Trong code:
```python
loss = MSELoss(current_q, td_target)
```

### Phân tích:

| Phần | Công thức | Ý nghĩa |
|---|---|---|
| **Prediction** | `Q_θ(s, a)` | Q-value mà mạng **dự đoán** cho (s, a) |
| **Target** | `r + γ · max_a' Q_θ(s', a')` | Q-value mà mạng **nên** output (theo Bellman) |
| **Error** | `target - prediction` | TD error (δ) |
| **Loss** | `error²` | Bình phương error → MSE |

### Quá trình tối ưu:
1. Tính loss L(θ).
2. Tính gradient ∂L/∂θ bằng backpropagation.
3. Cập nhật weights: θ ← θ - lr · ∂L/∂θ.
4. Lặp lại → L giảm dần → Q_θ(s, a) → giá trị đúng.

### Lưu ý quan trọng:
- Target cũng phụ thuộc θ → target **thay đổi** khi θ thay đổi → không ổn định.
- Giải pháp (DRL nâng cao): dùng **target network** (θ⁻ cố định) → ổn định hơn.

---

## Câu 8: Vì sao target có dạng max Q(s', a')?

```
target = r + γ · max_a' Q(s', a')
```

### Lý do:

1. **Bellman Optimality Equation**:
   ```
   Q*(s, a) = E[r + γ · max_a' Q*(s', a')]
   ```
   Target được xây dựng theo phương trình Bellman tối ưu — Q* thỏa mãn phương trình này.

2. **max**: Chọn **action tốt nhất** tại state tiếp theo.
   - Giả sử agent sẽ hành động **tối ưu** trong tương lai.
   - Đây là tính chất **off-policy** của Q-Learning.

3. **Off-policy learning**:
   - **Behavior policy**: ε-greedy (để explore).
   - **Target policy**: greedy (max Q) → policy tối ưu.
   - Q-Learning học Q* bất kể agent thực sự hành động thế nào.

4. **Hội tụ về Q***:
   - Nếu update liên tục theo max Q → Q hội tụ về Q* (Bellman optimality).

---

## Câu 9: Điều gì xảy ra nếu target thay đổi quá nhanh? So sánh Q-network với Q-table

### Target thay đổi quá nhanh:

**Vấn đề**:
```
target = r + γ · max_a' Q_θ(s', a')
                         ↑
                    dùng cùng θ đang được cập nhật!
```

1. **Moving target problem**: Mỗi khi θ thay đổi, target cũng thay đổi → mạng "đuổi theo" target di chuyển → **không ổn định**, có thể diverge.

2. **Oscillation**: Q-values dao động mạnh, không hội tụ.

3. **Overestimation**: max Q thường bị overestimate → target quá cao → Q bị inflate.

**Giải pháp (trong DQN thực tế, Bài 5)**:
- **Target network**: Dùng θ⁻ (copy cũ, cập nhật chậm) để tính target.
- **Experience replay**: Phá vỡ tương quan → ổn định gradient.
- **Double DQN**: Giảm overestimation.

### So sánh Q-network vs Q-table:

| Tiêu chí | Q-network | Q-table |
|---|---|---|
| **Tổng quát hóa** | Tốt — state chưa thăm cũng có Q hợp lý | Không — state chưa thăm: Q = 0 |
| **Tốc độ học** | Chậm hơn (cần nhiều data, gradient descent) | Nhanh hơn (cập nhật trực tiếp) |
| **State liên tục** | Xử lý trực tiếp | Cần discretize |
| **Scalability** | Rất tốt | Kém |
| **Stability** | Kém hơn (moving target, gradient issues) | Tốt hơn (convergence guarantee) |
| **Memory** | O(|θ|) – cố định | O(|S| × |A|) — tăng theo state space |

---

## Câu 10: Quan sát đồ thị loss: khi nào model học tốt?

*(Xem đồ thị `bai3_3_training_results.png` và `bai3_3_loss_analysis.png`)*

### Dấu hiệu model học tốt:

1. **Loss giảm dần**: Từ giá trị cao → thấp, cho thấy Q-function đang hội tụ.

2. **Loss ổn định**: Sau khi giảm, loss dao động quanh mức thấp (không tăng lại).

3. **Không spike**: Loss không có đột biến lớn (nếu có → training không ổn định).

### Các giai đoạn điển hình:

| Giai đoạn | Loss | Ý nghĩa |
|---|---|---|
| **Đầu (0-100 ep)** | Cao, giảm nhanh | Network đang học policy cơ bản |
| **Giữa (100-500 ep)** | Giảm chậm | Fine-tuning, Q-values hội tụ |
| **Cuối (500+ ep)** | Thấp, dao động nhẹ | Network đã học xong, loss ổn định |

### Lưu ý:
- Loss = 0 không nhất thiết tốt — có thể **overfitting**.
- Loss dao động nhẹ là **bình thường** trong RL (khác supervised learning).
- Nếu loss tăng trở lại → **instability** (cần kiểm tra learning rate, target network).

---

## Câu 11: Quan sát reward: khi nào agent đạt policy tốt?

*(Xem đồ thị reward trong `bai3_3_training_results.png`)*

### Dấu hiệu agent đạt policy tốt:

1. **Reward tăng dần**: Từ thấp (~10-20 ban đầu) → cao (~200-500).
2. **Reward ổn định ở mức cao**: Sau khi đạt đỉnh, reward duy trì ổn định.
3. **CartPole**: Max reward = 500 (giới hạn episode). Agent tốt đạt ~400-500.

### Các giai đoạn:

| Giai đoạn | Reward | Agent đang làm gì |
|---|---|---|
| **Exploration (đầu)** | Thấp (~10-30) | Agent hành động gần ngẫu nhiên (ε cao) |
| **Learning (giữa)** | Tăng dần | Agent bắt đầu học cân bằng cây |
| **Exploitation (cuối)** | Cao, ổn định (~200-500) | Agent đã tìm được policy tốt |

### Khi nào agent chưa tốt:
- Reward **không tăng** sau nhiều episode → learning rate sai, kiến trúc mạng kém.
- Reward **tăng rồi giảm** → catastrophic forgetting, training instability.
- Reward **dao động mạnh** → ε quá cao hoặc replay buffer quá nhỏ.

### So sánh với Q-table:
- Q-network thường đạt reward **cao hơn** vì xử lý state liên tục trực tiếp.
- Q-table bị giới hạn bởi discretization → policy kém chính xác hơn.
