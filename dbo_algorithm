import random
import math

DEFAULT_POP = 30
DEFAULT_ITER = 200
DEFAULT_PROB_LOCAL = 0.7  # not used by paper, kept for backward compatibility

rng = random.Random(42)


def dbo_algorithm(tasks, vms,
                  population=DEFAULT_POP,
                  max_iter=DEFAULT_ITER,
                  prob_local=DEFAULT_PROB_LOCAL):

    n_tasks = len(tasks)
    n_vm = len(vms)

    if n_tasks == 0 or n_vm == 0:
        return {}

    vm_mips = [vm.mips for vm in vms]
    vm_names = [vm.name for vm in vms]

    # paper-matching parameters
    alpha = 0.6
    sigma = 0.4
    b_const = 0.5
    k_defect = 0.05
    S_steal = 0.5

    # operator probabilities
    p_roll = 0.25
    p_dance = 0.20
    p_breed = 0.20
    p_forage = 0.20
    p_steal = 0.15

    # -------------- fitness (round continuous to VM index) -----------------
    def fitness_from_cont(pos):
        vm_loads = [0.0] * n_vm
        for i, task in enumerate(tasks):
            idx = int(round(pos[i]))
            if idx < 0: idx = 0
            if idx >= n_vm: idx = n_vm - 1
            vm_loads[idx] += task.cpu_load / vm_mips[idx]
        return max(vm_loads)

    # ----------------- initialization ---------------------
    population_cont = []
    population_prev = []

    for _ in range(population):
        pos = [rng.uniform(0, n_vm - 1) for _ in range(n_tasks)]
        population_cont.append(pos)
        population_prev.append(pos.copy())

    # find initial best/worst
    best = None
    best_fit = float('inf')
    worst = None
    worst_fit = -float('inf')

    for pos in population_cont:
        f = fitness_from_cont(pos)
        if f < best_fit:
            best_fit = f
            best = pos.copy()
        if f > worst_fit:
            worst_fit = f
            worst = pos.copy()

    # ==============================================================
    #                        MAIN LOOP
    # ==============================================================

    for t in range(1, max_iter + 1):
        # shrinking factor for breeding/foraging region
        R = 1.0 - (t / float(max_iter))

        # recompute global best & worst each iteration
        for pos in population_cont:
            f = fitness_from_cont(pos)
            if f < best_fit:
                best_fit = f
                best = pos.copy()
            if f > worst_fit:
                worst_fit = f
                worst = pos.copy()

        # vectors used by all operators
        X_best = best.copy()      # Xb
        X_worst = worst.copy()    # Xw
        X_star = X_best           # X* = global optimum (correct)

        # ----------------------------------------------------------
        #              iterate individuals
        # ----------------------------------------------------------
        for i in range(population):
            xi = population_cont[i]
            xi_prev = population_prev[i]

            r = rng.random()
            if r < p_roll:
                op = "roll"
            elif r < p_roll + p_dance:
                op = "dance"
            elif r < p_roll + p_dance + p_breed:
                op = "breed"
            elif r < p_roll + p_dance + p_breed + p_forage:
                op = "forage"
            else:
                op = "steal"

            new_x = xi.copy()

            # ------------------------------------------------------
            # 1. BALL-ROLLING  (Eq.2)
            # ------------------------------------------------------
            if op == "roll":
                for d in range(n_tasks):
                    def_term = alpha * k_defect * (xi[d] - xi_prev[d])
                    Dx = abs(xi[d] - X_worst[d])
                    new_x[d] = xi[d] + def_term + b_const * Dx

            # ------------------------------------------------------
            # 2. DANCING  (Eq.3)
            # ------------------------------------------------------
            elif op == "dance":
                theta = rng.uniform(0, math.pi)
                tan_th = math.tan(theta)
                for d in range(n_tasks):
                    new_x[d] = xi[d] + tan_th * abs(xi[d] - xi_prev[d])

            # ------------------------------------------------------
            # 3. BREEDING (Eq.4 - Eq.5)
            # ------------------------------------------------------
            elif op == "breed":
                Lb, Ub = 0.0, float(n_vm - 1)
                Lb_star = [max(X_star[d] * (1.0 - R), Lb) for d in range(n_tasks)]
                Ub_star = [min(X_star[d] * (1.0 + R), Ub) for d in range(n_tasks)]

                b1 = [rng.random() for _ in range(n_tasks)]
                b2 = [rng.random() for _ in range(n_tasks)]

                for d in range(n_tasks):
                    new_x[d] = (
                        X_star[d]
                        + b1[d] * (xi[d] - Lb_star[d])
                        + b2[d] * (xi[d] - Ub_star[d])
                    )

                    if new_x[d] < Lb_star[d]:
                        new_x[d] = Lb_star[d]
                    if new_x[d] > Ub_star[d]:
                        new_x[d] = Ub_star[d]

            # ------------------------------------------------------
            # 4. FORAGING (Eq.6 - Eq.7)
            # ------------------------------------------------------
            elif op == "forage":
                Lb, Ub = 0.0, float(n_vm - 1)
                Lbb = [max(X_best[d] * (1.0 - R), Lb) for d in range(n_tasks)]
                Ubb = [min(X_best[d] * (1.0 + R), Ub) for d in range(n_tasks)]

                for d in range(n_tasks):
                    C1 = rng.gauss(0, 1)
                    C2 = rng.random()
                    new_x[d] = xi[d] + C1 * (xi[d] - Lbb[d]) + C2 * (xi[d] - Ubb[d])
                    if new_x[d] < Lbb[d]:
                        new_x[d] = Lbb[d]
                    if new_x[d] > Ubb[d]:
                        new_x[d] = Ubb[d]

            # ------------------------------------------------------
            # 5. STEALING (Eq.8)
            # ------------------------------------------------------
            elif op == "steal":
                other_idx = rng.randrange(population)
                X_bj = population_cont[other_idx]

                for d in range(n_tasks):
                    g = rng.random()
                    term = abs(xi[d] - X_star[d]) + abs(xi[d] - X_bj[d])
                    new_x[d] = X_best[d] + S_steal * g * term

            # ------------------------------------------------------
            #  EXTRA SMALL MIX OF ORIGINAL ALGORITHM 1  (optional)
            # ------------------------------------------------------
            for d in range(n_tasks):
                alg1 = alpha * (X_best[d] - xi[d]) + sigma * (rng.uniform(0, n_vm - 1) - xi[d])
                new_x[d] += 0.1 * alg1

            # clamp to continuous bounds
            for d in range(n_tasks):
                if new_x[d] < 0.0: new_x[d] = 0.0
                if new_x[d] > n_vm - 1: new_x[d] = float(n_vm - 1)

            # accept if improved
            if fitness_from_cont(new_x) < fitness_from_cont(xi):
                population_prev[i] = xi.copy()
                population_cont[i] = new_x
            else:
                population_prev[i] = xi.copy()

        # end population loop
    # end iter loop

    # ----------------- produce assignment -------------------
    assignment = {}
    for i, val in enumerate(best):
        vm_idx = int(round(val))
        if vm_idx < 0: vm_idx = 0
        if vm_idx >= n_vm: vm_idx = n_vm - 1
        assignment[tasks[i].id] = vm_names[vm_idx]

    print(f"DBO (full 5-operator): Optimized makespan = {best_fit:.4f} s")
    return assignment
