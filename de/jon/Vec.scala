//class Vec(val array: Array[Double]) extends AnyVal {
//
//  def size = array.length
//  def apply(i: Int) = array(i)
//
//  def update(i: Int, value: Double) = array(i) = value
//
//  def +(that: Vec) = Vec.add(this, that)
//  def -(that: Vec) = Vec.sub(this, that)
//  def *(scalar: Double) = Vec.mul(this, scalar)
//
//  def maximum(min: Double) = Vec.vecSOp(this, min, (x, min) => math.max(x, min))
//  def sum = array.sum
//
//  override def toString: String = "[" + array.map(e => f"${e}%.3f").mkString(" ") + "]"
//
//}
//
//object Vec {
//
//  def apply(size: Int) = new Vec(new Array[Double](size))
//  def apply(array: Array[Double]) = new Vec(array)
//  def unapplySeq(v : Vec) = Array.unapplySeq(v.array)
//
//  def elemOp(a: Vec, b: Vec, op: (Double, Double) => Double) = {
//    var i = 0
//    val result = Vec(a.size)
//    while (i < a.size) {
//      result(i) = op(a(i), b(i))
//      i += 1
//    }
//    result
//  }
//
//  def vecSOp(a: Vec, s: Double, op: (Double, Double) => Double) = {
//    var i = 0
//    val result = Vec(a.size)
//    while (i < a.size) {
//      result(i) = op(a(i), s)
//      i += 1
//    }
//    result
//  }
//
//  def add(a: Vec, b: Vec) = elemOp(a, b,_ + _)
//  def sub(a: Vec, b: Vec) = elemOp(a, b, _ - _)
//  def mul(a: Vec, s: Double) = vecSOp(a, s, _ * _)
//
//}
object VecOps {

  type Vec = Array[Double]

  implicit class VecOpsClass(val array: Vec) extends AnyVal {
      def +(that: Vec) = add(array, that)
      def -(that: Vec) = sub(array, that)
      def *(scalar: Double) = mul(array, scalar)
  }

  def elemOp2(a: Vec, b: Vec, op: (Double, Double) => Double) = {
    var i = 0
    val result = new Array[Double](a.length)
    while (i < a.size) {
      result(i) = op(a(i), b(i))
      i += 1
    }
    result
  }

  def elemOp(a: Vec, op: (Double) => Double) = {
    var i = 0
    val result = new Array[Double](a.length)
    while (i < a.size) {
      result(i) = op(a(i))
      i += 1
    }
    result
  }

  def add(a: Vec, b: Vec) = elemOp2(a, b,_ + _)
  def sub(a: Vec, b: Vec) = elemOp2(a, b, _ - _)
  def mul(a: Vec, s: Double) = elemOp(a, _ * s)

}
