// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package rewrite

import (
	"errors"
	"github.com/jonlawlor/matrixexp"
	"reflect"
)

// A Compiler can transform matrix expressions by applying various optimizations
// to it, such as converting a.T().Mul(b.T()) to an appropriate call to DGEMM,
// or allowing concurrent evaluation of (a.Mul(b)).Add(c.Mul(d)) via Async.
type Compiler interface {
	// Compile transforms a matrix, or returns an error indicating a problem
	// with either the matrix expression or the compiler.
	Compile(matrixexp.MatrixExp) (matrixexp.MatrixExp, error)

	// MustCompile is just like Compile, except that instead of returning an error
	// it will panic if it encounters a problem.  It is intended for use in init()
	MustCompile(matrixexp.MatrixExp) matrixexp.MatrixExp
}

// Rewriter can convert one matrix expression to another.
type Rewriter interface {
	Rewrite(matrixexp.MatrixExp) (matrixexp.MatrixExp, error)
}

// A Matcher can determine if an expression wildcard matches another matrix
// expression.
type Matcher interface {
	Match(matrixexp.MatrixExp) bool
}

// template represents an example based rule for the compiler to follow.
type template struct {
	from matrixexp.MatrixExp
	to   matrixexp.MatrixExp
}

// Template produces a rewrite rule from a matrix expression template.
func Template(from, to matrixexp.MatrixExp) Rewriter {
	return &template{
		from: from,
		to:   to,
	}
}

// Rewrite applies a rewrite rule to a matrix expression.  If it can't be applied
// to the input expression, then it returns an error.
func (r *template) Rewrite(m1 matrixexp.MatrixExp) (matrixexp.MatrixExp, error) {

	// Determine if the matrix expression matches the rewrite rule.
	// TODO(jonlawlor): implement some kind of reflection cache.
	m2 := m1.Copy()
	matMap := make(map[matrixexp.MatrixExp]matrixexp.MatrixExp)
	if ok := matches(m1, r.from, matMap); !ok {
		// TODO(jonlawlor): implement useful error message.  This is just a placeholder.
		// We'll probably need a better way of converting a matrix expression to a
		// string.
		return nil, errors.New("expression does not match")
	}
	return m2, nil
}

// matches returns true if the matrix expression matches the template.  It also
// modifies the map to contain the mapping between the input matrix expression
// and the template output.
func matches(m1, from matrixexp.MatrixExp, matMap map[matrixexp.MatrixExp]matrixexp.MatrixExp) bool {
	// jonlawlor: would this be better returning an error?
	if w, ok := from.(Matcher); ok {
		// from is a wildcard, so it can be directly compared
		if !w.Match(m1) {
			return false
		}
		// Determine if we have seen the expression before.
		if to, seen := matMap[from]; !seen {
			matMap[from] = m1
		} else if to != m1 {
			return false
		}
		// I'm not sure if a wlldcard should have subexpressions.  For now we'll
		// assume no.
		return true
	}
	// Determine if m1 and from are the same matrix operation.
	rm1 := reflect.ValueOf(m1)
	rfrom := reflect.ValueOf(from)
	if rm1.Type() != rfrom.Type() {
		return false
	}
	// Walk subexpressions.
	rm1 = follow(rm1)
	rfrom = follow(rfrom)
	for i := 0; i < rfrom.NumField(); i++ {
		// if rfrom is a matrix expression, call matches on it as well
		if f := rfrom.Field(i); f.Type().Implements(reflect.TypeOf((*matrixexp.MatrixExp)(nil)).Elem()) {
			if fieldMatch := matches(f.Interface().(matrixexp.MatrixExp), rm1.Field(i).Interface().(matrixexp.MatrixExp), matMap); !fieldMatch {
				return false
			}
		}
	}
	return true
}

func follow(r1 reflect.Value) reflect.Value {
	for ; r1.Kind() == reflect.Ptr; r1 = r1.Elem() {
	}
	return r1
}
