# Bài 3.1 - Temporal Difference Learning: Trả lời câu hỏi

---

## Câu 1: Bootstrapping trong TD Learning là gì? Vì sao TD không cần đợi đến cuối episode?

**Bootstrapping** là kỹ thuật cập nhật giá trị ước lượng dựa trên **giá trị ước lượng khác** (chưa chính xác hoàn toàn), thay vì đợi kết quả thực tế.

Trong TD(0), TD target là:
```
TD target = r + γ·V(s')
```

Ở đây, `V(s')` là **ước lượng** giá trị trạng thái tiếp theo — không phải giá trị thực. Chính việc dùng `V(s')` (một ước lượng) để cập nhật `V(s)` (ước lượng khác) được gọi là **bootstrapping**.

**Vì sao TD không cần đợi đến cuối episode?**
- TD chỉ cần quan sát **một bước**: `(s, a, r, s')` rồi cập nhật ngay.
- Monte Carlo phải đợi tới cuối episode để tính return `G = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...`.
- TD thay thế phần "tương lai chưa biết" bằng `V(s')`, cho phép cập nhật trực tuyến (online).

---

## Câu 2: So sánh cách cập nhật của TD(0) với Monte Carlo Prediction

| Tiêu chí | TD(0) | Monte Carlo |
|---|---|---|
| **Công thức** | `V(s) ← V(s) + α[r + γ·V(s') - V(s)]` | `V(s) ← V(s) + α[G - V(s)]` |
| **Target** | `r + γ·V(s')` (bootstrap) | `G` (actual return) |
| **Thời điểm cập nhật** | Sau mỗi bước (online) | Sau khi kết thúc episode |
| **Số bước cần** | 1 bước | Toàn bộ episode |
| **Bootstrapping** | Có (dùng V(s') ước lượng) | Không (dùng G thực tế) |
| **Bias** | Có (do bootstrap) | Không (unbiased) |
| **Variance** | Thấp hơn | Cao hơn |

**Tóm tắt:**
- TD(0) dùng ước lượng 1 bước: chỉ cần `r + γ·V(s')`.
- MC dùng toàn bộ return thực tế `G` từ trạng thái hiện tại đến cuối episode.

---

## Câu 3: TD(0) có bias và variance như thế nào so với Monte Carlo?

### Bias:
- **TD(0) có bias**: Do dùng `V(s')` (giá trị ước lượng, chưa chính xác) làm target. Nếu `V(s')` sai, update sẽ sai theo → biased.
- **MC không có bias (unbiased)**: Do dùng actual return `G` — là giá trị thực nhận được từ episode.

### Variance:
- **TD(0) có variance thấp**: Chỉ phụ thuộc vào 1 reward `r` và 1 ước lượng `V(s')`.
- **MC có variance cao**: Return `G = r_t + γr_{t+1} + γ²r_{t+2} + ...` phụ thuộc vào **tất cả** reward trong episode. Nhiều biến ngẫu nhiên → variance tích lũy.

### Kết luận:
- TD(0): **biased, low variance** → hội tụ nhanh hơn trong thực tế.
- MC: **unbiased, high variance** → hội tụ chậm hơn nhưng chính xác hơn khi đủ dữ liệu.

---

## Câu 4: Vì sao TD Learning phù hợp với môi trường online hơn Monte Carlo?

1. **Cập nhật từng bước**: TD cập nhật `V(s)` ngay sau mỗi transition `(s, a, r, s')`, không cần đợi episode kết thúc.
2. **Không cần episode kết thúc**: Trong môi trường continuing (không có terminal state), MC không thể hoạt động vì không bao giờ có `G` hoàn chỉnh. TD vẫn hoạt động bình thường.
3. **Phản hồi nhanh**: Agent học và cải thiện policy nhanh hơn vì update diễn ra liên tục.
4. **Tiết kiệm bộ nhớ**: TD không cần lưu toàn bộ episode.
5. **Phù hợp real-time**: Trong các ứng dụng thực tế (robot, game, trading), agent cần học ngay lập tức.

---

## Câu 5: Giải thích ý nghĩa từng thành phần trong công thức cập nhật TD(0)

```
V(s) ← V(s) + α · [r + γ·V(s') - V(s)]
```

| Thành phần | Ý nghĩa |
|---|---|
| `V(s)` | Giá trị ước lượng hiện tại của trạng thái `s` |
| `α` (alpha) | Learning rate — tốc độ học. Quyết định mức độ tin tưởng vào thông tin mới vs. thông tin cũ |
| `r` | Reward tức thời nhận được khi chuyển từ `s` sang `s'` |
| `γ` (gamma) | Discount factor — hệ số chiết khấu. Thể hiện mức độ "quan tâm" đến phần thưởng tương lai (0 ≤ γ ≤ 1) |
| `V(s')` | Giá trị ước lượng của trạng thái tiếp theo (bootstrap estimate) |
| `r + γ·V(s')` | **TD Target** — mục tiêu mà V(s) nên tiến tới |
| `r + γ·V(s') - V(s)` | **TD Error (δ)** — sự khác biệt giữa target và ước lượng hiện tại. Phản ánh "sự bất ngờ" |
| `α · δ` | Bước cập nhật thực tế: V(s) dịch chuyển một phần (α) theo hướng giảm TD error |

---

## Câu 6: Nếu learning rate quá lớn hoặc quá nhỏ thì ảnh hưởng gì đến hội tụ?

### α quá lớn (ví dụ α = 0.5 hoặc 1.0):
- V(s) thay đổi **quá mạnh** mỗi bước → dao động mạnh (oscillation).
- Có thể **không hội tụ** hoặc hội tụ rất không ổn định.
- Mỗi lần update ghi đè phần lớn giá trị cũ → quên thông tin đã học.
- **Hiệu ứng**: Learning curve dao động nhiều, không mượt.

### α quá nhỏ (ví dụ α = 0.001):
- V(s) thay đổi **quá chậm** → hội tụ cực kỳ chậm.
- Cần rất nhiều episode để V(s) phản ánh giá trị đúng.
- Tuy nhiên, khi hội tụ sẽ **ổn định hơn**.
- **Hiệu ứng**: Learning curve rất phẳng, tiến bộ chậm.

### α phù hợp (ví dụ α = 0.1):
- Cân bằng giữa tốc độ học và độ ổn định.
- Hội tụ nhanh, learning curve mượt.

---

## Câu 7: Khi gamma tiến gần 1, giá trị trạng thái thay đổi như thế nào?

- **γ → 1**: Agent quan tâm đến phần thưởng **xa trong tương lai** nhiều hơn.
  - V(s) tích lũy nhiều reward hơn → **giá trị V(s) lớn hơn**.
  - Trong CartPole (reward = +1 mỗi bước), V(s) ≈ tổng số bước trung bình từ s đến terminal.
  - Ví dụ: Nếu episode trung bình 200 bước, V(s_start) ≈ 200.

- **γ → 0**: Agent chỉ quan tâm đến reward **tức thời**.
  - V(s) ≈ r (chỉ reward bước tiếp theo).
  - Trong CartPole: V(s) ≈ 1 cho mọi state (vì reward mỗi bước = 1).

- **Ảnh hưởng thực tế**:
  - γ lớn → agent có tầm nhìn dài → quyết định tốt hơn nhưng hội tụ chậm hơn.
  - γ nhỏ → agent "cận thị" → hội tụ nhanh nhưng có thể không optimal.

---

## Câu 8: Cài đặt bảng giá trị V(s) cho môi trường CartPole như thế nào khi state là continuous?

CartPole có state liên tục 4 chiều → **không thể trực tiếp dùng bảng** (vô hạn trạng thái).

### Giải pháp: Discretization (Rời rạc hóa)

1. **Xác định khoảng giá trị hợp lý** cho từng chiều:
   - Cart position: [-2.4, 2.4]
   - Cart velocity: [-3.0, 3.0] (clipped)
   - Pole angle: [-0.21, 0.21]
   - Angular velocity: [-3.0, 3.0] (clipped)

2. **Chia mỗi chiều thành N bins** (ví dụ N = 10):
   ```python
   bins = np.linspace(low, high, N+1)[1:-1]  # N-1 điểm chia → N bins
   ```

3. **Chuyển state liên tục → tuple chỉ số**:
   ```python
   def discretize(state, bins):
       return tuple(np.digitize(state[i], bins[i]) for i in range(4))
   ```

4. **Dùng dictionary làm bảng V(s)**:
   ```python
   V = defaultdict(float)  # V[(2, 5, 4, 7)] = 0.0
   ```

### Lưu ý:
- Số bins quá nhỏ → mất thông tin, không phân biệt được state.
- Số bins quá lớn → bảng quá lớn (N^4), cần nhiều dữ liệu.
- N = 10 → 10,000 states — hợp lý cho CartPole.

---

## Câu 9: Viết hàm update TD(0) và giải thích từng dòng code

```python
def td0_update(V, state, reward, next_state, done, alpha=0.1, gamma=0.99):
    """
    Cập nhật V(s) theo TD(0).
    """
    # Tính TD target: nếu done thì V(s') = 0
    if done:
        td_target = reward  # Không có tương lai nữa
    else:
        td_target = reward + gamma * V[next_state]  # Bootstrap: dùng V(s') ước lượng
    
    # Tính TD error: sai lệch giữa target và ước lượng hiện tại
    td_error = td_target - V[state]
    
    # Cập nhật V(s): dịch chuyển V(s) một phần (alpha) về phía target
    V[state] = V[state] + alpha * td_error
    
    return td_error
```

**Giải thích từng dòng:**

| Dòng | Giải thích |
|---|---|
| `if done: td_target = reward` | Nếu episode kết thúc, không có state tiếp theo → target chỉ là reward cuối |
| `td_target = reward + gamma * V[next_state]` | TD target = reward tức thời + γ × giá trị ước lượng tương lai |
| `td_error = td_target - V[state]` | TD error = (target) - (ước lượng hiện tại). Nếu > 0: V(s) quá thấp, cần tăng |
| `V[state] += alpha * td_error` | Cập nhật V(s): nếu error > 0 → tăng V(s), nếu error < 0 → giảm V(s) |

---

## Câu 10: Làm thế nào để discretize state trong CartPole?

### Bước 1: Xác định bounds cho mỗi feature

```python
STATE_BOUNDS = [
    (-2.4, 2.4),    # cart position
    (-3.0, 3.0),    # cart velocity (clip vì lý thuyết là [-∞, +∞])
    (-0.21, 0.21),  # pole angle (~12°)
    (-3.0, 3.0),    # angular velocity (clip)
]
```

### Bước 2: Tạo bins

```python
NUM_BINS = 10
bins = []
for low, high in STATE_BOUNDS:
    bins.append(np.linspace(low, high, NUM_BINS + 1)[1:-1])
    # Ví dụ: 10 bins → 9 điểm chia
```

### Bước 3: Discretize

```python
def discretize_state(state, bins):
    discrete = []
    for i, val in enumerate(state):
        clipped = np.clip(val, STATE_BOUNDS[i][0], STATE_BOUNDS[i][1])
        idx = np.digitize(clipped, bins[i])  # Trả về chỉ số bin (0 đến NUM_BINS-1)
        discrete.append(idx)
    return tuple(discrete)  # Ví dụ: (3, 5, 4, 7)
```

### Kết quả:
- Mỗi state liên tục → tuple 4 số nguyên → dùng làm key cho dictionary.
- Tổng: 10^4 = 10,000 trạng thái rời rạc.

---

## Câu 11: So sánh TD(0) với Monte Carlo theo: tốc độ hội tụ, độ ổn định

| Tiêu chí | TD(0) | Monte Carlo |
|---|---|---|
| **Tốc độ hội tụ** | Nhanh hơn — cập nhật mỗi bước, bootstrapping lan truyền thông tin sớm | Chậm hơn — phải đợi cuối episode, variance cao gây dao động |
| **Độ ổn định** | Ổn định hơn với α phù hợp — variance thấp vì chỉ phụ thuộc (r, V(s')) | Kém ổn định hơn — G phụ thuộc nhiều biến ngẫu nhiên |
| **Bias** | Có bias (bootstrap) nhưng giảm dần khi học | Không bias |
| **Với episode dài** | Tốt — không bị ảnh hưởng bởi độ dài episode | Kém — variance tăng theo độ dài episode |
| **Nhạy cảm với α** | Vừa phải | Ít hơn (vì dùng incremental mean) |

**Kết quả thí nghiệm (từ chương trình):**
- TD(0) thường cho learning curve **mượt hơn** và **hội tụ nhanh hơn**.
- MC có learning curve **dao động nhiều hơn** nhưng không bị bias.

---

## Câu 12: Khi số episode tăng, sai số giữa V(s) và giá trị thực thay đổi ra sao?

- **Ban đầu**: V(s) = 0 cho mọi state → sai số lớn.
- **Giai đoạn đầu (0-1000 episodes)**: Sai số giảm nhanh khi agent khám phá và cập nhật nhiều state.
- **Giai đoạn giữa (1000-3000)**: Sai số giảm chậm hơn, V(s) bắt đầu hội tụ.
- **Giai đoạn cuối (3000+)**: Sai số dao động quanh mức thấp — không về 0 hoàn toàn do:
  - Discretization gây mất thông tin.
  - Random policy không tối ưu.
  - Learning rate cố định (α = 0.1).

**Xu hướng tổng quát**: Sai số giảm theo dạng **hàm mũ giảm** (exponential decay), nhanh ban đầu, chậm dần về sau.

---

## Câu 13: Vẽ learning curve và giải thích xu hướng

*(Xem đồ thị trong file `bai3_1_td_vs_mc.png` sau khi chạy chương trình)*

### Giải thích xu hướng:

1. **TD Error curve**: Giảm dần theo episode, cho thấy V(s) dần hội tụ. Đầu tiên error cao vì V(s) = 0, sau đó giảm khi ước lượng chính xác hơn.

2. **Reward curve**: Với random policy, reward trung bình dao động quanh ~20-25 (CartPole với random policy). Không tăng vì policy không được cải thiện (chỉ làm prediction).

3. **So sánh TD vs MC**: TD error thường giảm mượt hơn MC error, phản ánh variance thấp hơn của TD(0).

---

## Câu 14: TD(0) có thể áp dụng trực tiếp cho action-value không?

**Có**, nhưng cần điều chỉnh:

Thay vì V(s), ta ước lượng Q(s, a):

```
Q(s, a) ← Q(s, a) + α · [r + γ · Q(s', a') - Q(s, a)]
```

Đây chính là thuật toán **SARSA** (State-Action-Reward-State-Action):
- On-policy: dùng action a' thực sự được chọn tại s'.
- Cần quan sát cả (s, a, r, s', a').

Biến thể **off-policy** là **Q-Learning**:
```
Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') - Q(s, a)]
```
- Dùng `max Q(s', a')` thay vì Q(s', a'): tách policy học (greedy) và policy khám phá (ε-greedy).

---

## Câu 15: Nếu reward rất thưa (sparse), TD(0) gặp vấn đề gì?

### Vấn đề:
1. **Lan truyền chậm**: TD(0) chỉ truyền thông tin 1 bước. Nếu reward chỉ xuất hiện ở bước cuối (ví dụ: +1 khi thắng game), thông tin reward phải "chảy ngược" qua nhiều bước mới ảnh hưởng đến V(s) ban đầu.
   - Cần nhiều episode để V(s) ở các state xa reward hội tụ.

2. **Hầu hết TD error ≈ 0**: Ngoại trừ state gần reward, các state khác có `r = 0` và `V(s') ≈ 0` → TD error nhỏ → V(s) gần như không đổi.

3. **Hội tụ chậm**: Agent cần nhiều episode để thông tin reward lan truyền đến các state xa.

### Giải pháp:
- **n-step TD**: Truyền thông tin xa hơn mỗi bước (→ Bài 3.2).
- **Eligibility Traces (TD(λ))**: Kết hợp nhiều n-step → lan truyền nhanh hơn.
- **Reward shaping**: Thêm reward trung gian để hướng dẫn agent.
- **Monte Carlo**: Không bị vấn đề này vì dùng actual return.
