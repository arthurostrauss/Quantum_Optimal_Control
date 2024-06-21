// OpenQASM 3.0 standard gate library
//
// Note: the gates defined by this file are exactly the set that the OpenQASM
// 3.0 specification defined, as are the unitary actions, but implementations
// have some scope for how they handle the file internally.  See the
// `source/language/standard_library.rst` documentation for full detail.


// Pauli gate: bit-flip or NOT gate
gate x a { U(π, 0, π) a;}
// Pauli gate: bit and phase flip
gate y a { U(π, π/2, π/2) a;}


