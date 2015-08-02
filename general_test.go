// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"math/rand"
	"testing"
)

// A MatrixFixture represents a test fixture for a matrix expression.
type MatrixFixture struct {
	name string         // the name of the fixture
	r, c int            // expected size of the result
	expr MatrixExpr     // expression being tested
	want blas64.General // expected result of the expression
}

// UnaryExpr and BinaryExpr are functions used to construct matrix fixtures
// from other fixtures.  Matrix algebra will generally be a Unary operation
// (such as transpose, or asynchronous operations) or a binary operation (such
// as Add, Mul, etc.)
type UnaryExpr func(MatrixFixture) *MatrixFixture
type BinaryExpr func(MatrixFixture, MatrixFixture) *MatrixFixture

// Functions to help generate example matrix literals.
func GeneralZeros(r, c int) MatrixExpr {
	return &General{zeros(r, c)}
}
func zeros(r, c int) blas64.General {
	return blas64.General{
		Rows:   r,
		Cols:   c,
		Stride: c,
		Data:   make([]float64, r*c),
	}
}

func GeneralOnes(r, c int) MatrixExpr {
	return &General{ones(r, c)}
}
func ones(r, c int) blas64.General {
	m := zeros(r, c)
	for i := range m.Data {
		m.Data[i] = 1
	}
	return m
}

func GeneralRand(r, c int) MatrixExpr {
	return &General{rnd(r, c)}
}
func rnd(r, c int) blas64.General {
	m := zeros(r, c)
	rs := rand.New(rand.NewSource(99))
	for i := range m.Data {
		m.Data[i] = rs.NormFloat64()
	}
	return m
}

func eye(r int) blas64.General {
	g := zeros(r, r)
	for i := 0; i < r*r; i += r + 1 {
		g.Data[i] = 1
	}
	return g
}

// blasadd uses GEMM to add two matrices for comparison.
func blasadd(g1, g2 blas64.General) blas64.General {
	// first make a copy of g1
	g3 := blas64.General{
		Rows:   g1.Rows,
		Cols:   g1.Cols,
		Stride: g1.Stride,
		Data:   make([]float64, len(g1.Data)),
	}
	for i, v := range g1.Data {
		g3.Data[i] = v
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1.0, eye(g1.Rows), g2, 1.0, g3)
	return g3
}

// blasmul uses GEMM to multiply two matrices for comparison.
func blasmul(g1, g2 blas64.General) blas64.General {
	// first the receiver
	g3 := blas64.General{
		Rows:   g1.Rows,
		Cols:   g2.Cols,
		Stride: g2.Stride,
		Data:   make([]float64, g1.Rows*g2.Stride),
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1.0, g1, g2, 0.0, g3)
	return g3
}

// blassub uses GEMM to subtract one matrix from another for comparison.
func blassub(g1, g2 blas64.General) blas64.General {
	// first make a copy of g1
	g3 := blas64.General{
		Rows:   g1.Rows,
		Cols:   g1.Cols,
		Stride: g1.Stride,
		Data:   make([]float64, len(g1.Data)),
	}
	for i, v := range g1.Data {
		g3.Data[i] = v
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1.0, eye(g1.Rows), g2, 1.0, g3)
	return g3
}

// blastrans uses GEMM to transpose a matrix.
func blastrans(g1 blas64.General) blas64.General {
	// first the receiver
	g3 := blas64.General{
		Rows:   g1.Cols,
		Cols:   g1.Rows,
		Stride: g1.Rows,
		Data:   make([]float64, g1.Rows*g1.Cols),
	}
	blas64.Gemm(blas.NoTrans, blas.Trans, 1.0, eye(g1.Cols), g1, 0.0, g3)
	return g3
}

// {Op}Generator functions take a Matrix Fixture (or pair of fixtures) and if
// the operation can be performed on the two fixtures, then it creates a new
// fixture by applying the operation to both the matrix expression and the
// expected result.  If the given operation cannot be applied, then it returns
// nil.

// AddGenerator creates an Add expression.
func AddGenerator(a, b MatrixFixture) *MatrixFixture {
	if a.r != b.r || a.c != b.c {
		return nil
	}

	m := &MatrixFixture{
		name: "(" + a.name + ") + (" + b.name + ")",
		r:    a.r,
		c:    a.c,
		expr: a.expr.Add(b.expr),
		want: blasadd(a.want, b.want),
	}
	return m
}

// AsyncGenerator creates an Async expression.
func AsyncGenerator(a MatrixFixture) *MatrixFixture {
	m := &MatrixFixture{
		name: "Async(" + a.name + ")",
		r:    a.r,
		c:    a.c,
		expr: &Async{a.expr},
		want: a.want,
	}
	return m
}

// DivElemGenerator creates a DivElem expression.
func DivElemGenerator(a, b MatrixFixture) *MatrixFixture {
	if a.r != b.r || a.c != b.c {
		return nil
	}

	// avoid div zero
	wantData := make([]float64, len(a.want.Data))
	for i, v := range b.want.Data {
		if v == 0.0 {
			return nil
		}
		wantData[i] = a.want.Data[i] / v
	}

	m := &MatrixFixture{
		name: "(" + a.name + ") ./ (" + b.name + ")",
		r:    a.r,
		c:    a.c,
		expr: a.expr.DivElem(b.expr),
		want: blas64.General{
			Rows:   a.r,
			Cols:   a.c,
			Stride: a.c,
			Data:   wantData,
		},
	}
	return m
}

// FutureGenerator creates a Future expression.
func FutureGenerator(a MatrixFixture) *MatrixFixture {
	m := &MatrixFixture{
		name: "Future(" + a.name + ")",
		r:    a.r,
		c:    a.c,
		expr: NewFuture(a.expr),
		want: a.want,
	}
	return m
}

// MulGenerator creates a Mul expression.
func MulGenerator(a, b MatrixFixture) *MatrixFixture {
	if a.c != b.r {
		return nil
	}

	m := &MatrixFixture{
		name: "(" + a.name + ") * (" + b.name + ")",
		r:    a.r,
		c:    b.c,
		expr: a.expr.Mul(b.expr),
		want: blasmul(a.want, b.want),
	}
	return m
}

// MulElemGenerator creates a MulElem expression.
func MulElemGenerator(a, b MatrixFixture) *MatrixFixture {
	if a.r != b.r || a.c != b.c {
		return nil
	}

	wantData := make([]float64, len(a.want.Data))
	for i, v := range b.want.Data {
		wantData[i] = a.want.Data[i] * v
	}

	m := &MatrixFixture{
		name: "(" + a.name + ") .* (" + b.name + ")",
		r:    a.r,
		c:    a.c,
		expr: a.expr.MulElem(b.expr),
		want: blas64.General{
			Rows:   a.r,
			Cols:   a.c,
			Stride: a.c,
			Data:   wantData,
		},
	}
	return m
}

// SubGenerator creates a Sub expression.
func SubGenerator(a, b MatrixFixture) *MatrixFixture {
	if a.r != b.r || a.c != b.c {
		return nil
	}

	m := &MatrixFixture{
		name: "(" + a.name + ") - (" + b.name + ")",
		r:    a.r,
		c:    a.c,
		expr: a.expr.Sub(b.expr),
		want: blassub(a.want, b.want),
	}
	return m
}

// TransposeGenerator creates a transpose expression.
func TransposeGenerator(a MatrixFixture) *MatrixFixture {

	m := &MatrixFixture{
		name: "(" + a.name + ")'",
		r:    a.c,
		c:    a.r,
		expr: a.expr.T(),
		want: blastrans(a.want),
	}
	return m
}

// GeneralMatrices are a set of example matrix literals for use in generating
// more complicated expressions.
var GeneralMatrices = []MatrixFixture{
	{name: "General 1x1 zeros", r: 1, c: 1, expr: GeneralZeros(1, 1), want: zeros(1, 1)},
	{name: "General 10x1 zeros", r: 10, c: 1, expr: GeneralZeros(10, 1), want: zeros(10, 1)},
	{name: "General 1x10 zeros", r: 1, c: 10, expr: GeneralZeros(1, 10), want: zeros(1, 10)},
	{name: "General 10x10 zeros", r: 10, c: 10, expr: GeneralZeros(10, 10), want: zeros(10, 10)},

	{name: "General 1x1 ones", r: 1, c: 1, expr: GeneralOnes(1, 1), want: ones(1, 1)},
	{name: "General 10x1 ones", r: 10, c: 1, expr: GeneralOnes(10, 1), want: ones(10, 1)},
	{name: "General 1x10 ones", r: 1, c: 10, expr: GeneralOnes(1, 10), want: ones(1, 10)},
	{name: "General 10x10 ones", r: 10, c: 10, expr: GeneralOnes(10, 10), want: ones(10, 10)},

	{name: "General 1x1 normal rand", r: 1, c: 1, expr: GeneralRand(1, 1), want: rnd(1, 1)},
	{name: "General 10x1 normal rand", r: 10, c: 1, expr: GeneralRand(10, 1), want: rnd(10, 1)},
	{name: "General 1x10 normal rand", r: 1, c: 10, expr: GeneralRand(1, 10), want: rnd(1, 10)},
	{name: "General 10x10 normal rand", r: 10, c: 10, expr: GeneralRand(10, 10), want: rnd(10, 10)},
}

// UnaryGenerators are the set of unary expressions to be applied to mutate the
// set of test fixtures.
var UnaryGenerators = []UnaryExpr{
	AsyncGenerator,
	FutureGenerator,
	TransposeGenerator,
}

// UnaryGenerators are the set of binary expressions to be applied to mutate the
// set of test fixtures.
var BinaryGenerators = []BinaryExpr{
	AddGenerator,
	DivElemGenerator,
	MulGenerator,
	MulElemGenerator,
	SubGenerator,
}

// TestFixtures is the set of expressions to test.
var TestFixtures []MatrixFixture

// MutateFixtures takes a set of fixtures, and then adds to them by applying
// various possible operations.  This is kind of like a production rule, cool.
func MutateFixtures(fix1, fix2 []MatrixFixture) []MatrixFixture {
	var f []MatrixFixture

	for _, f1 := range fix1 {
		for _, gen := range UnaryGenerators {
			fixture := gen(f1)
			if fixture != nil {
				f = append(f, *fixture)
			}
		}
		for _, f2 := range fix2 {
			for _, gen := range BinaryGenerators {
				fixture := gen(f1, f2)
				if fixture != nil {
					f = append(f, *fixture)
				}
			}
		}
	}
	return f
}

// init sets the TestFixtures variable to cover the complete set of possible
// matrix expressions.
func init() {
	f := GeneralMatrices
	for level := 0; level < 1; level++ {
		mf := MutateFixtures(f, f)
		f = append(f, mf...)
	}
	TestFixtures = f

}

// TestGeneral evaluates matrix expressions to compare their output with the
// expected output.
func TestGeneral(t *testing.T) {
	for i, tt := range TestFixtures {
		m := tt.expr
		r, c := m.Dims()
		if r != tt.r {
			t.Errorf("%d: %s rows are %d, want %d", i, tt.name, r, tt.r)
		}
		if c != tt.c {
			t.Errorf("%d: %s columns are %d, want %d", i, tt.name, c, tt.c)
		}
		want := &General{tt.want}
		if got := m.Eval(); !Equals(got, want) {
			t.Errorf("%d: %s equals %v, want %v", i, tt.name, got, want)
		}
	}
}
