import itertools


class Calculator:
    """Basic calculator tool for Ritsu."""


    def calculate(self, expression: str) -> float:
        """Evaluate a simple math expression safely."""
        try:
            # Only allow numbers and math operators
            allowed = "0123456789+-*/(). "
            if not all(c in allowed for c in expression):
                raise ValueError("Invalid characters in expression")
            return eval(expression, {"__builtins__": {}})
        except Exception as e:
            return f"Error: {e}"  
        

    def sum(self, a: float, b: float) -> float:
        return a + b
    def subtract(self, a: float, b: float) -> float:
        return a - b
    def multiply(self, a: float, b: float) -> float:
        return a * b
    def divide(self, a: float, b: float) -> float:
        if b == 0:
            return "Error: Division by zero"
        return a / b
    def power(self, a: float, b: float) -> float:
        return a ** b
    def sqrt(self, a: float,) -> float:
        if a < 0:
            return "Error: Square root of negative number"
        return a ** 0.5
    def divide_percentage(self, a: float, b: float) -> float:
        if b == 0:
            return "Error: Division by zero"
        return (a / b) * 100
    def percentage_of(self, a: float, b: float) -> float:
        return (a / 100) * b


    def solve_2d_lp(constraints, objective):
        """
        constraints: list of tuples representing boundary equations and their operator.
        Each constraint is a dict:
            {'type':'bound', 'var':index, 'op':'<= or >=', 'value':v}  OR
            {'type':'line', 'coeffs':(a,b), 'op':'<= or >=', 'value':c}  which means a*x + b*y <= c  (or >=)
        objective: tuple of coefficients (cx, cy) to minimize cx*x + cy*y
        Works only for 2 variables (x,y).
        """
        # Convert >= constraints to <= by flipping sign when helpful for generating equalities.
        # We'll collect boundary lines as equalities (a*x + b*y = c) to compute intersections.
        lines = []  # (a,b,c) representing a*x + b*y = c (these are boundaries)
        bounds = {} # variable bounds: {0: (min,max), 1: (min,max)} use None for inf
        
        # Duyệt qua tất cả các ràng buộc trong danh sách
        for con in constraints:
            
            # Nếu ràng buộc là dạng "bound" (giới hạn trên/dưới cho 1 biến x hoặc y)
            if con['type'] == 'bound' and con['var'] in (0,1):
                idx = con['var']   # idx = 0 nếu là biến x, idx = 1 nếu là biến y
                op = con['op']     # toán tử: "<=" hoặc ">="
                v = con['value']   # giá trị biên (số thực)
                
                # Lấy khoảng hiện tại (lower, upper) của biến từ bounds, 
                # nếu chưa có thì mặc định là (None, None)
                cur = bounds.get(idx, (None, None))
                
                # Nếu là ràng buộc dạng "≤"
                if op == '<=':
                    # cập nhật upper bound (giá trị trên) cho biến
                    bounds[idx] = (cur[0], v if cur[1] is None else min(cur[1], v))
                    # thêm đường biên ứng với ràng buộc này: x = v hoặc y = v
                    lines.append((1 if idx==0 else 0, 1 if idx==1 else 0, v))  
                
                # Nếu là ràng buộc dạng "≥"
                elif op == '>=':
                    # cập nhật lower bound (giá trị dưới) cho biến
                    bounds[idx] = (v if cur[0] is None else max(cur[0], v), cur[1])
                    # thêm đường biên ứng với ràng buộc này: x = v hoặc y = v
                    lines.append((1 if idx==0 else 0, 1 if idx==1 else 0, v))

                elif op == '=':
                # ràng buộc bằng -> tức là cả lower và upper bound đều bằng v
                    bounds[idx] = (v, v)
                    lines.append((1 if idx==0 else 0, 1 if idx==1 else 0, v))
                
                elif op == '<':
        # coi như <=
                    bounds[idx] = (cur[0], v if cur[1] is None else min(cur[1], v))
                    lines.append((1 if idx==0 else 0, 1 if idx==1 else 0, v))

                elif op == '>':
                    # coi như >=
                    bounds[idx] = (v if cur[0] is None else max(cur[0], v), cur[1])
                    lines.append((1 if idx==0 else 0, 1 if idx==1 else 0, v))
            
            # Nếu ràng buộc là dạng "line" (bất đẳng thức tổng quát a*x + b*y ≤ c hoặc ≥ c)
            elif con['type'] == 'line':
                a, b = con['coeffs']   # hệ số (a, b) của biến
                op = con['op']         # toán tử: "<=" hoặc ">="
                c = con['value']       # hằng số vế phải
                
                # lưu đường biên a*x + b*y = c
                lines.append((a, b, c))
            
            # Nếu không thuộc dạng nào trong 2 loại trên -> báo lỗi
            else:
                raise ValueError("Định dạng ràng buộc không được hỗ trợ")


        # Build all candidate vertices by intersecting every pair of boundary lines
        def intersect(l1, l2):
            a1,b1,c1 = l1
            a2,b2,c2 = l2
            det = a1*b2 - a2*b1
            if abs(det) < 1e-12:
                return None
            x = (c1*b2 - c2*b1) / det
            y = (a1*c2 - a2*c1) / det
            return (x,y)

        candidates = set()
        # Add intersections of lines
        for l1, l2 in itertools.combinations(lines, 2):
            p = intersect(l1, l2)
            if p:
                candidates.add((round(p[0],12), round(p[1],12)))
        # Also consider variable bound extremes if present
        # For each var bound, combine with other boundaries (or with other bound)
        for idx in (0,1):
            bnd = bounds.get(idx)
            if bnd:
                lo, hi = bnd
                for v in (lo, hi):
                    if v is None: 
                        continue
                    # fix variable idx = v and intersect with all lines
                    for l in lines:
                        a,b,c = l
                        if idx == 0:
                            # x = v => a*v + b*y = c => y = (c - a*v)/b if b != 0
                            if abs(b) > 1e-12:
                                y = (c - a*v)/b
                                candidates.add((round(v,12), round(y,12)))
                        else:
                            # y = v => a*x + b*v = c => x = (c - b*v)/a if a != 0
                            if abs(a) > 1e-12:
                                x = (c - b*v)/a
                                candidates.add((round(x,12), round(v,12)))
        # Filter feasible candidates by all constraints
        def check_point(pt):
            x,y = pt
            # variable bounds
            bx = bounds.get(0,(None,None))
            by = bounds.get(1,(None,None))
            if bx[0] is not None and x < bx[0] - 1e-9: return False
            if bx[1] is not None and x > bx[1] + 1e-9: return False
            if by[0] is not None and y < by[0] - 1e-9: return False
            if by[1] is not None and y > by[1] + 1e-9: return False
            # line constraints: must satisfy original operators; we assume all provided constraints are enforced already by lines,
            # but to be safe, evaluate them too (caller may extend)
            for con in constraints:
                if con['type'] == 'line':
                    a,b = con['coeffs']
                    op = con['op']
                    c = con['value']
                    lhs = a*x + b*y
                    if op == '<=' and lhs > c + 1e-9: return False
                    if op == '>=' and lhs < c - 1e-9: return False
            return True

        feasible = [pt for pt in candidates if check_point(pt)]
        if not feasible:
            return None, None

        # evaluate objective
        cx, cy = objective
        best = min(feasible, key=lambda p: cx*p[0] + cy*p[1])
        best_val = cx*best[0] + cy*best[1]
        return best, best_val

    # Build constraints for the example:
    constraints = [
        {'type':'bound', 'var':1, 'op':'>=', 'value':0},   # y >= 0
        {'type':'bound', 'var':1, 'op':'<=', 'value':5},   # y <= 5
        {'type':'bound', 'var':0, 'op':'>=', 'value':0},   # x >= 0
        {'type':'line', 'coeffs':(1,1), 'op':'>=', 'value':2},  # x + y >= 2
        {'type':'line', 'coeffs':(1,-1), 'op':'<=', 'value':2}, # x - y <= 2
    ]

    objective = (1.0, -2.0)  # minimize x - 2y

    best_point, best_val = solve_2d_lp(constraints, objective)
    print("Best point:", best_point)
    print("Best value:", best_val)


    def pythagorean(a: float, b: float) -> float:
        """Return the length of the hypotenuse of a right triangle given sides a and b."""
        return (a**2 + b**2) ** 0.5
    

    def factorial(self, n: int) -> int:
        """Return the factorial of n."""
        if n < 0:
            return "Error: Factorial of negative number"
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result













if __name__ == "__main__":
    # Hàm mục tiêu: f(x,y) = x - 2y
    def obj(x, y): return x - 2*y

    # bounds: [(lb_x, ub_x), (lb_y, ub_y)]
    bounds = [(0, None), (0, 5)]  # x >=0, 0 <= y <= 5

    # lines: (a, b, c, op) = a*x + b*y (<= or >=) c
    lines = [
        (1, 1, 2, ">="),   # x + y >= 2
        (1, -1, 2, "<="),  # x - y <= 2
    ]

    val, pt = Calculator.solve_2d_lp(bounds, lines, obj, minimize=True)
    print("Giá trị tối ưu:", val)
    print("Tại điểm:", pt)

