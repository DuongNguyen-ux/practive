# Bài 3.2 - n-step Q-Learning: Trả lời câu hỏi

---

## Câu 1: Ý tưởng chính của n-step learning là gì?

**n-step learning** kết hợp ưu điểm của **TD(0)** và **Monte Carlo**:

- **TD(0)**: Chỉ nhìn 1 bước → bias cao nhưng variance thấp.
- **Monte Carlo**: Nhìn toàn bộ episode → unbiased nhưng variance cao.
- **n-step**: Nhìn **n bước** → cân bằng bias-variance.

**Ý tưởng**: Thay vì chỉ dùng 1 reward + bootstrap (TD) hoặc tất cả reward (MC), ta dùng **n reward thực tế** rồi mới bootstrap.

```
n = 1:  G = r + γ·max Q(s', a)                    ← TD(0)
n = 3:  G = r + γr' + γ²r'' + γ³·max Q(s''', a)   ← n-step
n = ∞:  G = r + γr' + γ²r'' + ...                  ← Monte Carlo
```

**Lợi ích**: Lan truyền thông tin reward nhanh hơn TD(0), nhưng ổn định hơn MC.

---

## Câu 2: n-step return khác gì so với return của TD(0) và Monte Carlo?

| Phương pháp | Return | Bootstrap? | Số reward thực |
|---|---|---|---|
| **TD(0)** | `G = r + γ·max Q(s', a)` | Có, tại bước 1 | 1 |
| **n-step** | `G = r + γr' + ... + γ^{n-1}r^{(n-1)} + γ^n·max Q(s^{(n)}, a)` | Có, tại bước n | n |
| **Monte Carlo** | `G = r + γr' + γ²r'' + ... + γ^{T-1}r^{(T-1)}` | Không | Toàn bộ (T bước) |

**Sự khác biệt chính:**
- TD(0): Bootstrap ngay sau 1 bước → nhanh nhưng phụ thuộc nhiều vào ước lượng Q.
- n-step: Dùng n reward thực trước khi bootstrap → thông tin chính xác hơn.
- MC: Toàn bộ return thực → chính xác nhất nhưng phải đợi cuối episode.

---

## Câu 3: Khi n tăng, bias và variance thay đổi như thế nào?

### Bias:
- **n nhỏ (gần TD)**: Bias **cao** vì bootstrap sớm với Q ước lượng chưa chính xác.
- **n lớn (gần MC)**: Bias **giảm** vì dùng nhiều reward thực hơn, ít phụ thuộc ước lượng.
- **n = ∞ (MC)**: Bias = 0 (unbiased).

### Variance:
- **n nhỏ**: Variance **thấp** vì chỉ phụ thuộc ít biến ngẫu nhiên.
- **n lớn**: Variance **tăng** vì phụ thuộc nhiều reward (nhiều biến ngẫu nhiên hơn).
- **n = ∞ (MC)**: Variance **cao nhất**.

### Tóm tắt:
```
n tăng:  Bias ↓ , Variance ↑
n giảm:  Bias ↑ , Variance ↓
```

**Kết luận**: Cần chọn n phù hợp để cân bằng bias-variance tradeoff. Thường n = 3-5 cho kết quả tốt.

---

## Câu 4: Giải thích công thức n-step return trong Q-learning

```
G_n = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{n-1}·r_{t+n-1} + γ^n · max_a Q(s_{t+n}, a)
```

### Phân tích từng phần:

| Phần | Ý nghĩa |
|---|---|
| `r_t` | Reward tức thời tại bước t |
| `γ·r_{t+1}` | Reward bước t+1 đã chiết khấu |
| `γ²·r_{t+2}` | Reward bước t+2 đã chiết khấu 2 lần |
| `γ^{n-1}·r_{t+n-1}` | Reward cuối cùng trong chuỗi n bước |
| `γ^n · max_a Q(s_{t+n}, a)` | **Bootstrap term**: giá trị ước lượng từ bước n trở đi |

### Cách hoạt động:
1. **n bước đầu**: Dùng reward **thực tế** đã chiết khấu.
2. **Sau n bước**: Bootstrap bằng Q-value ước lượng (max vì Q-learning là off-policy).
3. **Nếu episode kết thúc sớm** (bước k < n): `G = r_t + γr_{t+1} + ... + γ^{k-1}r_{t+k-1}` (không bootstrap).

---

## Câu 5: Vai trò của max Q trong cập nhật là gì?

```
max_a Q(s_{t+n}, a)
```

**Vai trò:**
1. **Bootstrap estimate**: Ước lượng giá trị tương lai từ bước n trở đi mà không cần chờ episode kết thúc.
2. **Greedy selection**: `max` chọn action **tốt nhất** tại state `s_{t+n}`, tức target policy là **greedy** → tạo ra **off-policy** learning.
3. **Optimistic update**: Luôn giả sử agent sẽ chọn hành động tốt nhất trong tương lai → giúp hội tụ về Q* (Q-value tối ưu).

**So sánh:**
- Q-Learning: `max_a Q(s', a)` → off-policy (target policy ≠ behavior policy)
- SARSA: `Q(s', a')` (a' thực tế) → on-policy

---

## Câu 6: Tại sao n-step giúp lan truyền reward nhanh hơn TD(0)?

### Ví dụ minh họa:
Xét episode 10 bước, reward chỉ có ở bước cuối (sparse reward):
```
Bước:  0  1  2  3  4  5  6  7  8  9
Reward: 0  0  0  0  0  0  0  0  0  +1
```

**TD(0) (n=1)**:
- Episode 1: Chỉ state 8 được cập nhật (nhận reward +1 trực tiếp).
- Episode 2: State 7 được cập nhật (thông qua V(state 8)).
- Episode 3: State 6...
- **Cần ~10 episodes** để thông tin đến state 0.

**3-step**:
- Episode 1: State 6, 7, 8 đều được cập nhật (n-step return cover 3 bước).
- Episode 2: State 3, 4, 5 có thể được cập nhật.
- **Cần ~3-4 episodes** để thông tin lan đến state 0.

### Kết luận:
- n-step truyền thông tin reward **xa hơn n lần** so với TD(0) mỗi episode.
- Đặc biệt hiệu quả với **sparse reward**.

---

## Câu 7: Thiết kế cấu trúc n-step buffer gồm những thành phần nào?

```python
class NStepBuffer:
    def __init__(self, n):
        self.n = n                          # Số bước lưu trữ
        self.states = deque(maxlen=n)       # Các trạng thái
        self.actions = deque(maxlen=n)      # Các hành động
        self.rewards = deque(maxlen=n)      # Các reward
```

### Các thành phần:

| Thành phần | Kiểu | Mục đích |
|---|---|---|
| `n` | int | Số bước n-step return |
| `states` | deque(maxlen=n) | Lưu n trạng thái gần nhất |
| `actions` | deque(maxlen=n) | Lưu n hành động tương ứng |
| `rewards` | deque(maxlen=n) | Lưu n reward tương ứng |

### Phương thức cần có:
1. `add(state, action, reward)` — Thêm transition mới
2. `is_full()` — Kiểm tra đã đủ n bước chưa
3. `get_nstep_return(gamma, Q, next_state, done)` — Tính n-step return
4. `reset()` — Xóa buffer khi bắt đầu episode mới

### Tại sao dùng `deque(maxlen=n)`?
- Tự động loại bỏ phần tử cũ nhất khi thêm phần tử mới → **sliding window** tự nhiên.
- Hiệu suất O(1) cho thêm/xóa ở hai đầu.

---

## Câu 8: Làm thế nào để xử lý khi episode kết thúc trước khi đủ n bước?

Khi episode kết thúc (done = True) mà buffer chưa đủ n bước, ta cần **flush buffer**:

```python
# Sau khi episode kết thúc, xử lý các bước còn trong buffer
while len(buffer.states) > 0:
    # Tính return cho k bước còn lại (k < n)
    G = 0
    for i, r in enumerate(buffer.rewards):
        G += (gamma ** i) * r
    # KHÔNG bootstrap vì episode đã kết thúc (done = True)
    
    s0, a0 = buffer.states[0], buffer.actions[0]
    Q[(s0, a0)] += alpha * (G - Q[(s0, a0)])
    
    # Bỏ phần tử đầu tiên
    buffer.states.popleft()
    buffer.actions.popleft()
    buffer.rewards.popleft()
```

### Nguyên lý:
- Khi episode kết thúc, không còn `V(s')` để bootstrap → tính return thuần túy (giống MC).
- Các state-action còn trong buffer vẫn phải được cập nhật, nếu không sẽ bỏ lỡ thông tin.

---

## Câu 9: Viết hàm tính n-step return từ buffer

```python
def compute_nstep_return(buffer, gamma, Q, next_state, done, num_actions=2):
    """
    Tính n-step return từ buffer.
    
    G_n = Σ_{i=0}^{n-1} γ^i · r_{t+i} + γ^n · max_a Q(s_{t+n}, a)
    """
    G = 0.0
    
    # Phần 1: Tổng discounted rewards trong buffer
    for i, reward in enumerate(buffer.rewards):
        G += (gamma ** i) * reward
        # i=0: G += r_t
        # i=1: G += γ·r_{t+1}
        # i=2: G += γ²·r_{t+2}
        # ...
    
    # Phần 2: Bootstrap value (chỉ khi chưa kết thúc)
    if not done:
        # max_a Q(s_{t+n}, a): chọn action tốt nhất tại state sau n bước
        max_q = max(Q[(next_state, a)] for a in range(num_actions))
        G += (gamma ** len(buffer.rewards)) * max_q
        # γ^n · max Q — chiết khấu n bước rồi bootstrap
    
    return G
```

**Giải thích:**
1. **Vòng for**: Tích lũy reward thực với discount tăng dần (`γ^0, γ^1, γ^2, ...`).
2. **if not done**: Nếu episode chưa kết thúc, thêm phần bootstrap.
3. **max_q**: Lấy Q-value cao nhất tại state tiếp theo (greedy — tính chất off-policy).
4. **gamma^n**: Chiết khấu phần bootstrap n bước.

---

## Câu 10: So sánh với các giá trị n = 1, 3, 5: tốc độ học, độ ổn định

*(Kết quả cụ thể phụ thuộc vào thí nghiệm. Dưới đây là xu hướng điển hình)*

| Tiêu chí | n = 1 (TD) | n = 3 | n = 5 |
|---|---|---|---|
| **Tốc độ học ban đầu** | Chậm | Trung bình | Nhanh |
| **Độ ổn định** | Cao | Trung bình | Thấp hơn |
| **Reward cuối** | Tốt | Thường tốt nhất | Có thể dao động |
| **Variance** | Thấp | Trung bình | Cao hơn |
| **Lan truyền reward** | Chậm (1 bước/ep) | Nhanh hơn 3× | Nhanh hơn 5× |

### Nhận xét:
- **n = 1**: Ổn định nhưng chậm, đặc biệt với sparse reward.
- **n = 3**: Thường là **sweet spot** — cân bằng tốt giữa tốc độ và ổn định.
- **n = 5**: Học nhanh nhưng learning curve có thể dao động nhiều hơn.

---

## Câu 11: Khi n quá lớn, điều gì xảy ra?

1. **Variance tăng mạnh**: Return phụ thuộc nhiều biến ngẫu nhiên → ước lượng không ổn định.
2. **Tiệm cận Monte Carlo**: Khi n ≥ độ dài episode, n-step trở thành MC → mất lợi thế TD.
3. **Delay cập nhật**: Cần đợi n bước mới update → phản hồi chậm.
4. **Memory tăng**: Buffer phải lưu n transitions.
5. **Nhạy cảm với stochasticity**: Trong môi trường ngẫu nhiên, nhiều bước = nhiều "noise" tích lũy.

### Ví dụ:
- n = 100 trong CartPole (max 500 bước): Gần MC, variance cao, học chậm.
- n = 500: Hoàn toàn MC → phải đợi cuối episode.

---

## Câu 12: So sánh n-step Q-learning với TD(0)

| Tiêu chí | TD(0) Q-Learning | n-step Q-Learning |
|---|---|---|
| **Return** | 1-step: `r + γ·max Q` | n-step: `r + γr' + ... + γ^n·max Q` |
| **Bias** | Cao hơn | Thấp hơn |
| **Variance** | Thấp hơn | Cao hơn |
| **Lan truyền reward** | Chậm (1 bước/episode) | Nhanh (n bước/episode) |
| **Sparse reward** | Kém | Tốt hơn |
| **Đơn giản** | Đơn giản hơn | Phức tạp hơn (cần buffer) |
| **Memory** | O(1) per transition | O(n) per buffer |
| **Tốc độ hội tụ** | Tùy bài toán | Thường nhanh hơn với n phù hợp |

**Kết luận**: n-step thường **tốt hơn TD(0)** nhưng cần chọn n cẩn thận.

---

## Câu 13: Khi nào nên chọn n nhỏ? Khi nào nên chọn n lớn?

### Nên chọn n nhỏ (n = 1-3):
- Môi trường có **reward dày đặc** (dense reward) — không cần lan truyền xa.
- Môi trường có **nhiều tính ngẫu nhiên** (stochastic) — giảm variance.
- Cần **ổn định** hơn tốc độ.
- Episode **ngắn** — n lớn không có ý nghĩa.

### Nên chọn n lớn (n = 5-10+):
- Môi trường có **reward thưa** (sparse reward) — cần lan truyền nhanh.
- Môi trường **deterministic** — variance do n lớn không ảnh hưởng nhiều.
- Episode **dài** — n lớn giúp bootstrap từ ước lượng xa hơn.
- Cần **hội tụ nhanh** hơn **ổn định**.

### Thực tế:
- n = 3-5 thường là lựa chọn tốt cho hầu hết bài toán.
- Có thể thử nhiều giá trị n và chọn bằng cross-validation.

---

## Câu 14: n-step có phù hợp với môi trường dài episode không?

### Trả lời: **Có**, nhưng cần lưu ý:

**Ưu điểm với episode dài:**
1. **Hiệu quả hơn MC**: MC phải đợi cuối episode (rất lâu). n-step cập nhật sau n bước.
2. **Lan truyền nhanh hơn TD(0)**: Trong episode 1000 bước, TD(0) cần ~1000 episode để thông tin từ cuối về đầu. n=10 chỉ cần ~100 episode.
3. **Không phụ thuộc độ dài episode**: Luôn dùng đúng n bước, không bị ảnh hưởng bởi T.

**Lưu ý:**
1. **Chọn n << T** (T = độ dài episode): Nếu n quá gần T, gần MC.
2. **Buffer management**: n lớn cần nhiều memory hơn nhưng vẫn O(n), không phải O(T).
3. **Stale buffer**: Nếu policy thay đổi nhanh, thông tin trong buffer có thể lỗi thời.

**Kết luận**: n-step đặc biệt **phù hợp** với episode dài — đó chính là ưu thế so với MC.
