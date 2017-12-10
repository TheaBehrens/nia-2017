//import VecOps.Vec
//import Vec.Vec
//import DE.Vec
//import DE.{Population, Vec}

object DE {

  import VecOps.VecOpsClass

  import scala.util.Random

  type Vec = Array[Double]

  def ackley(x: Vec) = {
    val a = 20.0
    val b = 0.2
    val c = 2.0 * math.Pi
    0.0
//    val ssq = x.map(e => math.pow(e, 2.0)).sum
//    val ssc = x.map(e => math.cos(c * e)).sum
//    -20.0 * math.exp(-b * math.sqrt(ssq / x.size)) - math.exp(ssc / x.size) + math.E + 20.0
  }

  def demand(price: Double, maxPrice: Double, maxDemand: Double) = {
    if (price > maxPrice) 0.0
    else if (price <= 0.0) maxDemand
    else maxDemand - math.pow(price, 2.0) * maxDemand / math.pow(maxPrice, 2.0)
  }

  def cost(x: Double, kwh: Double, cost: Double, max: Double) = {
    if (x <= 0.0) 0.0
    else if (x > kwh * max) Double.MaxValue
    else math.ceil(x / kwh) * cost
  }

  def profit2(x: Vec) = {
    val Array(e1, e2, e3, s1, s2, s3, p1, p2, p3) = VecOps.elemOp(x, math.max(_, 0.0))
    val d1 = demand(p1, 0.45, 2000000.0)
    val d2 = demand(p2, 0.25, 30000000.0)
    val d3 = demand(p3, 0.20, 20000000.0)
    val r1 = math.min(d1, s1) * p1
    val r2 = math.min(d2, s2) * p2
    val r3 = math.min(d3, s3) * p3
    val revenue = r1 + r2 + r3
    val c1 = cost(e1, 50000.0, 10000.0, 100)
    val c2 = cost(e2, 600000.0, 80000.0, 50)
    val c3 = cost(e3, 4000000.0, 400000.0, 3)
    val productionCost = c1 + c2 + c3
    val purchaseAmount = math.max((s1 + s2 + s3) - (e1 + e2 + e3), 0.0)
    val purchaseCost = purchaseAmount * 0.6
    val totalCost = productionCost + purchaseCost
    val profit = revenue - totalCost
    profit
  }

  def profit(x: Vec): Double = {
    //        val xp = x.maximum(0.0)
//    if (x.exists(_ < 0.0)) return Double.MinValue

    //VecOps.elemOp(x, math.max(_, 0.0)
    val Array(e1, e2, e3, st1, st2, st3, p1, p2, p3) = x
    val s1 = math.max(st1, 0.0)
    val s2 = math.max(st2, 0.0)
    val s3 = math.max(st3, 0.0)
    val d1 = demand(p1, 0.45, 2000000.0)
    val d2 = demand(p2, 0.25, 30000000.0)
    val d3 = demand(p3, 0.20, 20000000.0)
    val r1 = math.max(math.min(d1, s1), 0.0) * p1
    val r2 = math.max(math.min(d2, s2), 0.0) * p2
    val r3 = math.max(math.min(d3, s3), 0.0) * p3
    val revenue = r1 + r2 + r3
    val c1 = cost(e1, 50000.0, 10000.0, 100)
    val c2 = cost(e2, 600000.0, 80000.0, 50)
    val c3 = cost(e3, 4000000.0, 400000.0, 3)
    val productionCost = c1 + c2 + c3
    val purchaseAmount = math.max((s1 + s2 + s3) - (e1 + e2 + e3), 0.0)
    val purchaseCost = purchaseAmount * 0.6
    val totalCost = productionCost + purchaseCost
    val profit = revenue - totalCost
    //    println(s"revenue: $revenue, cost: $totalCost")
    //    if (totalCost < 0.00000001) {
    //
    //      println(
    //        f"e1: $e1%08.4e \t\t e2: $e2%08.4e \t\t e3: $e3%08.4e\n" +
    //        f"s1: $s1%08.4e \t\t s2: $s2%08.4e \t\t s3: $s3%08.4e\n" +
    //        f"p1: $p1%08.4f \t\t\t p2: $p2%08.4f \t\t\t p3: $p3%08.4f\n" +
    //        f"d1: $d1%08.4e \t\t d2: $d2%08.4e \t\t d3: $d3%08.4e\n" +
    //        f"r1: $r1%08.4e \t\t r2: $r2%08.4e \t\t r3: $r3%08.4e")
    //
    //      println(s"purchase amount: $purchaseAmount")
    //      println(s"production: $productionCost, purchase: $purchaseCost")
    //    }
    return profit
  }

  def de(population: Array[Vec], result: Array[Vec], Cr: Double, F: Double) = {
//    population
//      .map { target => target -> generateTrial(population, target, F, Cr) }
//      .map { case (target, trial) => if (profit(trial) > profit(target)) trial else target }
    var i = 0
    while (i < population.length) {
      val target = population(i)
      val trial = generateTrial(population, target, F, Cr)
      result(i) = if (profit(trial) > profit(target)) trial else target
      i += 1
    }
  }

  def generateTrial(population: Array[Vec], target: Vec, F: Double, Cr: Double) = {
    val Seq(base, x1, x2) = sample(population, n = 3)
    val donor = base + ((x1 - x2) * F)
    crossover(target, donor, Cr)
  }

  def crossover(a: Vec, b: Vec, Cr: Double) = {
    val r = Random.nextInt(a.size)
    val result = VecOps.elemOp2(a, b, (x, y) => if (Random.nextFloat > Cr) x else y)
    result(r) = b(r)
    result
  }

  def sample[T](elements: Seq[T]) = elements(Random.nextInt(elements.length))

  def sample[T](elements: Seq[T], n: Int) = Random.shuffle(elements).take(n)

  def deExperiment(popSize: Int, dim: Int, maxIterations: Int, initializer: Array[() => Double], F: Double, Cr: Double) = {
    var population = Array.fill(popSize)(Array.tabulate(dim) { i => initializer(i)() })//.map(Vec.apply)
    var nextPopulation = new Array[Vec](popSize)

    var i = 0
    var min = Double.MinValue
    while (i < maxIterations && min < 1514312.0) {
      de(population, nextPopulation, F = F, Cr = Cr)

      val temp = population
      population = nextPopulation
      nextPopulation = temp

      val bestSolution = population.maxBy(profit)
      val best = profit(bestSolution)
      if (best > min) {
        min = best
      }
      if (i % 1000 == 0)
        println(f"[$i%06d] best: $best%.2f")

      i += 1
    }
    min
  }

  def main(args: Array[String]) = {
    val popSize = 60
    val dim = 9
    val maxIterations = 50000
    val numTrials = 10

    val initializer = Array(
      () => Random.nextDouble()*50000.0*100.0,  // plant 1 max production
      () => Random.nextDouble()*600000.0*60.0,  // plant 2 ..
      () => Random.nextDouble()*4000000.0*3.0,  // plant 3 ..
      () => Random.nextDouble()*2000000.0,      // market 1 max demand
      () => Random.nextDouble()*30000000.0,     // market 2 ...
      () => Random.nextDouble()*20000000.0,     // market 3 ...
      () => Random.nextDouble()*0.45,           // market 1 max price
      () => Random.nextDouble()*0.25,           // market 2 ...
      () => Random.nextDouble()*0.2             // market 3 ...
    )

    val params =  for (f <- 0.1 until 1.0 by 0.1; cr <- 0.1 until 1.0 by 0.1) yield (f, cr)

//    val file = new PrintWriter("result.csv")
//    params.foreach( { case (f, cr) =>
      val f = 0.6
      val cr = 0.4
      var result = 0d
      var duration = 0l
      for (i <- 0 until numTrials) {
        val startTime = System.currentTimeMillis()
        result += deExperiment(popSize, dim, maxIterations, initializer, f, cr)
        duration += System.currentTimeMillis() - startTime
      }
      result = result / numTrials
      duration = duration / numTrials
      println(f"F=$f%.1f; Cr=$cr%.1f; mean result: $result%.2f; t = $duration ms")
      val line = f"$f%.1f;$cr%.1f;$result%.2f;$duration\n"
//      file.write(line)
//    })
//    file.close()

//    for (f <- 0.1 until 1.0 by 0.1; cr <- 0.1 until 1.0 by 0.1) {
//      val result = deExperiment(popSize, dim, maxIterations, initializer, f, cr)
//      println(f"best result for F=$f%.1f, Cr=$cr%.1f = $result%.2f")
//    }


  }

}

/*
F = 0.7, Cr = .5 works super fast
F = 0.7, Cr = 0.7
F = 0.4 works as well
[899000] best: 1514312,94
-3.4632057114966686E-6
6599999.999999953
1.1999999999997493E7
1063182.7883726095
1.1669156850702472E7
5867660.360914937
0.3079817245842124
0.19542071681638826
0.16812102568738288
 */