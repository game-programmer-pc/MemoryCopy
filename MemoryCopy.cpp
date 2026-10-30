#include "MemoryCopy.h"
#include <concepts>
#include <cassert>
#include <intrin.h>


#define ANONYMOUS_NAMESPACE_BEGIN namespace {
#define ANONYMOUS_NAMESPACE_END }

ANONYMOUS_NAMESPACE_BEGIN


/*

		[Microsoft] knows that user programs cannot modify control registers. This means that processor features cannot be changed after [OS] initialization.

		IsProcessorFeaturePresent is a [function] using the C calling convention, syntax, and rules but [it] still evaluates when [it] is called into a constant.
		[It] has no runtime overhead.


		[MSVC] does the work on the backend and returns a constant into the program. (If a different [compiler] is used,
		and [it] does not do this, the (MemoryCopy, procedure) is malformed. assert(!Procedure::MemoryCopy.malformed());.

		In C++20, functions are implicity constexpr. If the constructor was explicitly constexpr, the C API does not give C++ API enough information to express
		that the constexpr evaluation is a gauranteed promise. This would actually cause the compile time evaluation to fail.
		(   [IsProcessorFeaturePresent::Callee] is a C API Function that results in a constant, but
			[C++ constructors::Caller], if explicit, would say [it] fails the compile time condition
		)
		Leaving the constructor implicitly constexpr provides, in this case, a gaurantee due to differences in the two languages.


		m_StartingMemoryCopyIndex, using C++20, with the MEMORY_COPY::constructor being implicitly constexpr, can use a list of ternary operators to get a compile constant evaluation
		of what Processor Feature is present.

		C++20 also allows a member variable to be CONST as long as it is given a constant value during its constructor initializer list.

		m_StartingMemoryCopyIndex is used only in one place and it is read, not modified. The templated functions for AVX512, AVX2, and SSE2 will not be instantiated if the compiler
		determines the call site can never be executed.

		The class MEMORY_COPY is a singleton class with one static instance. If MemoryCopy is never called, no MEMORY_COPY class is ever instantiated
		and so there is no instance.

		If MemoryCopy is never called, MemoryCopy is deleted from the compiled program.
		The class MEMORY_COPY is never instantiated and so the static instance is never placed in data section(s).
		And the control paths that cannot be executed on certain CPUs (if the program is compiled on those CPUs) also do not get instantiated, including the paths templates.

		If concerned about a scenario of multiple threads trying to instantiate the class at the same time due to late initialization,
		it is not true late initialization. The compiler should determine that a call will be made to the MemoryCopy procedure and instantiate at compile time.
		No race condition. (MSVC does this. If a different compiler is used and does not do this, the MemoryCopy procedure is malformed).

		Do not go through a function pointer to ever call MemoryCopy (by grabbing the address of MemoryCopy).
		A late initialization method is being used, and the compiler does not instantiate templates unless used.
		The scenario being described here causes the compiler to have to trace, and it will cause the actual static MemoryCopy
		to not be optimized to ensure when the function pointer MemoryCopy is called, it does not break the program. (Instantiating all templates).

		GOOD:
			call MemoryCopy       ; Regular Microsoft X64 ABI Calling Convention.

		BAD:
			lea rax, MemoryCopy	  ; Load Effective Address of MemoryCopy.
								  ; Assume a different translation unit with different internal linkage.
								  ; RAX now has the MemoryCopy address.
								  ; The compiler no longer sees what is happening. It may begin template instantiation to avoid the program breaking.
								  ; It could also not do anything, making the MemoryCopy procedure malformed.
								  ; Also, this is now true late initialization, meaning threads could call it or attempt to at the same time, a race condition is now present.
								  ; Two static instances could be created or even more.

			mov rcx, [rax]		  ; The compiler cannot protect anything when a function address is grabbed. I have now dereferenced 8 bytes of instruction(s) from MemoryCopy.
								  ; There is no safe interface for this.
								  ; The templates are not instantiated, optimizations are not there, the compiler has no information, and the compiler also has no way to
								  ; to predict what is happening to protect from what is happening.

		To be clear, this is only unsafe because of templates because templates themselves are late initializations (not lazy).
		Late because there are not instantiated until they are called(first) and the initializations are the callees(last).

		The MemoryCopy procedure here may still be optimized, but that is because the method was designed for absolute zero runtime overhead,
		to avoid an EVENT 6 #UD fault.

		Other functions or procedures may see performance degradation and code bloat.

		This implementation of copying memory has no runtime overhead. You get exactly what the processor and OS allows.

*/


/*

	If the program that is going to be released when compiled needs to support a lower common denominator,
	the constructor for MEMORY_COPY can be modified to not test for certain processor features. Example shown nearby the MEMORY_COPY declaration [class, constructor] definition.

*/

#undef TRUE 
#undef FALSE

#define TRUE  true
#define FALSE false

typedef bool BOOLEAN;


template <typename T>
concept MEMORY_COPY_TYPE_ALIGNMENT = std::same_as<T, __m512i> || std::same_as<T, __m256i> || std::same_as<T, __m128i> ||
									 std::same_as<T, UINT64>  || std::same_as<T, UINT32>  || std::same_as<T, UINT16>  ||
									 std::same_as<T, UINT8>;

template <typename T>
concept MEMORY_COPY_VECTORIZED     = std::same_as<T, __m512i> || std::same_as<T, __m256i> || std::same_as<T, __m128i>;
								   
template <typename T>			   
concept MEMORY_COPY_SEQUENTIAL     = std::same_as<T, UINT64>  || std::same_as<T, UINT32>  || std::same_as<T, UINT16> || std::same_as<T, UINT8>;



class MEMORY_COPY_CONTEXT {
public:
	MEMORY_COPY_CONTEXT(
		LPCVOID		   Source,
		LPVOID		   Destination,
		SIZE_T		   Bytes
	) noexcept : m_Source(Source), m_Destination(Destination), m_Bytes(Bytes) {}

	LPCVOID m_Source;
	LPVOID  m_Destination;
	SIZE_T  m_Bytes;

};

enum class PROCESSOR_FEATURE : UINT64 {
	AVX512     = 0,
	AVX2       = 1,
	SSE2       = 2,
	SEQUENTIAL = 3
};

template <typename T>
requires MEMORY_COPY_TYPE_ALIGNMENT<T>
[[nodiscard]] constexpr SIZE_T CalculateTypeAlignmentShift() noexcept {
	
	if constexpr (std::same_as<T, __m512i>) {
		return static_cast<SIZE_T>(1) + CalculateTypeAlignmentShift<__m256i>();
	} else if constexpr (std::same_as<T, __m256i>) {
		return static_cast<SIZE_T>(1) + CalculateTypeAlignmentShift<__m128i>();
	} else if constexpr (std::same_as<T, __m128i>) {
		return static_cast<SIZE_T>(1) + CalculateTypeAlignmentShift<UINT64>();
	} else if constexpr (std::same_as<T, UINT64>) {
		return static_cast<SIZE_T>(1) + CalculateTypeAlignmentShift<UINT32>();
	} else if constexpr (std::same_as<T, UINT32>) {
		return static_cast<SIZE_T>(1) + CalculateTypeAlignmentShift<UINT16>();
	} else if constexpr (std::same_as<T, UINT16>) {
		return static_cast<SIZE_T>(1) + CalculateTypeAlignmentShift<UINT8>();
	} else if constexpr (std::same_as<T, UINT8>) {
		return static_cast<SIZE_T>(0);
	} else {
		static_assert("Unknown Type And Alignment");
	}
	
}

template <typename T>
requires MEMORY_COPY_VECTORIZED<T>
class VECTOR {
public:

	VECTOR() noexcept {
		
		if constexpr (std::same_as<T, __m512i>) {
			this->m_Vector = _mm512_setzero_si512();
		}
		else if constexpr (std::same_as<T, __m256i>) {
			this->m_Vector = _mm256_setzero_si256();
		}
		else if constexpr (std::same_as<T, __m128i>) {
			this->m_Vector = _mm_setzero_si128();
		} else {
			static_assert("Unknown vectorized type");
		}
	}

	VOID AlignedLoad(
		LPCVOID		Data
	) noexcept {

		if constexpr (std::same_as<T, __m512i>) {
			this->m_Vector = _mm512_load_si512(Data);
		}
		else if constexpr (std::same_as<T, __m256i>) {
			this->m_Vector = _mm256_load_si256(reinterpret_cast<CONST __m256i*>(Data));
		}
		else if constexpr (std::same_as<T, __m128i>) {
			this->m_Vector = _mm_load_si128(reinterpret_cast<CONST __m128i*>(Data));
		} else {
			static_assert("Unknown vectorized load intrinsic");
		}
	}


	VOID AlignedStore(
		LPVOID		 Data
	) noexcept {

		if constexpr (std::same_as<T, __m512i>) {
			
			_mm512_store_si512(
				Data,
				this->m_Vector
			);
		}
		else if constexpr (std::same_as<T, __m256i>) {
			
			_mm256_store_si256(
				reinterpret_cast<__m256i*>(Data),
				this->m_Vector
			);
		}
		else if constexpr (std::same_as<T, __m128i>) {
			
			_mm_store_si128(
				reinterpret_cast<__m128i*>(Data),
				this->m_Vector
			);

		} else {
			static_assert("Unknown vectorized store intrinsic");
		}
	}

private:

	T m_Vector;
};


/*
	When an unaligned read or write occurs, extra cycles are used. This is not just a performance degradation.
	It is a security vulnerability. Timing attacks occur when procedures (that are attempting to return the same result)
	do not complete at the same time as the previous procedure. 

	Unaligned vectorized register operations, causing cache line writeback operations through evictions, are the reason many game consoles are exploited.


	STARTING ADDRESS: 5
	BYTES			  64

	LEFT:  0-[B0,B1,B2,B3] 
	MIDDLE 5-[B4...B63]
	RIGHT  R-[B64...B127]

	This causes three cache line fetches. If shared, the core will generate a copy to its cache.
	This can cause data from the core's cache to be evicted. If the cache line to be evicted is shared, there is no write.
	But potentially, a cache line that is evicted must do a writeback operation. If it goes into L1, L2, or L3 cache, it is probably safe.
	But if it has to go to memory, the cache line is now able to be "snooped" from outside the processor.

	Since this is due to an unaligned load/store, the extra cycles provide a signature to trace, cache lines might be evicted from cache,
	since a cache writeback is 64 bytes, a hacker can make it seem to the processor as if its L1, L2, or L3 cache is smaller than it truly is,
	and cause consistent writeback evictions. 

	Since processes are file mapped, for paging, a page swap is copied into RAM. A video game executable itself can be larger than the amount of cache available
	across all cores and all caching levels. If unaligned vector load begins at the start of an executable, the processor cannot predict cache-line boundaries,
	causing many writeback operation evictions. 

	Essentially, where is the tail, body, and head? The processor attempts a load of the body, but it is not aligned, so it chases it tails. Then it resumes like a nerve signal
	up to its body, but it has lost sight of where it was going, so it needs its head. 

	X:192 bytes for unaligned compared to X:64 bytes for aligned. (CACHE LINE SIZE (X64) X: = 64).

	The sequence order for the cache line fetches is like: SECOND(LEFT)<-FIRST(MIDDLE)->THIRD(RIGHT).
	The sequence order could also be like: THIRD(LEFT)<-FIRST(MIDDLE)->SECOND(RIGHT).

	Whatever the sequence order is, the beginning is FIRST(MIDDLE).

*/

template <typename T>
requires MEMORY_COPY_VECTORIZED<T>
VOID MemoryCopyVectorized(
	MEMORY_COPY_CONTEXT& MemoryContext
) noexcept {

	SIZE_T	     Iterations						= MemoryContext.m_Bytes >> CalculateTypeAlignmentShift<T>();
	SIZE_T	     Iteration						= 0;
	CONST SIZE_T Mask							= alignof(T) - 1;
	CONST SIZE_T SourceAddressAlignment			= reinterpret_cast<UINT_PTR>(MemoryContext.m_Source)      & Mask;
	CONST SIZE_T DestinationAddressAlignment	= reinterpret_cast<UINT_PTR>(MemoryContext.m_Destination) & Mask;
	CONST SIZE_T ConsumedBytes					= sizeof(T) * Iterations;
	
	VECTOR<T> SourceVector;

	switch (SourceAddressAlignment) {
	case 0:
		switch (DestinationAddressAlignment) {
		case 0:
			break;
		default:
			return;
		}
	default:
		return;
	}



	for (;;) {

		switch (Iterations - Iteration) {
		
		case 0:

			MemoryContext.m_Source		= reinterpret_cast<LPCVOID>(reinterpret_cast<UINT_PTR>(MemoryContext.m_Source)     + ConsumedBytes);
			MemoryContext.m_Destination	= reinterpret_cast<LPVOID>(reinterpret_cast<UINT_PTR>(MemoryContext.m_Destination) + ConsumedBytes);
			MemoryContext.m_Bytes		-= ConsumedBytes;

			return;

		
		default:

			SourceVector.AlignedLoad(
				reinterpret_cast<CONST T*>(MemoryContext.m_Source) + Iteration
			);

			SourceVector.AlignedStore(
				reinterpret_cast<T*>(MemoryContext.m_Destination)  + Iteration
			);

			Iteration++;

		}
	}

}

template <typename T>
requires MEMORY_COPY_SEQUENTIAL<T>
VOID MemoryCopySequential(
	MEMORY_COPY_CONTEXT&  MemoryContext
) noexcept {

	SIZE_T       Iterations						= MemoryContext.m_Bytes >> CalculateTypeAlignmentShift<T>();
	SIZE_T       Iteration						= 0;
	CONST SIZE_T Mask						    = alignof(T) - 1;
	CONST SIZE_T SourceAddressAlignment			= reinterpret_cast<UINT_PTR>(MemoryContext.m_Source)	  & Mask;
	CONST SIZE_T DestinationAddressAlignment    = reinterpret_cast<UINT_PTR>(MemoryContext.m_Destination) & Mask;
	CONST SIZE_T ConsumedBytes					= sizeof(T) * Iterations;

	T Sequential;

	switch (SourceAddressAlignment) {
	case 0:
		switch (DestinationAddressAlignment) {
		case 0:
			break;
		default:
			return;
		}
	default:
		return;
	}

	for (;;) {
		
		switch (Iterations - Iteration) {
		case 0:
			
			MemoryContext.m_Source      = reinterpret_cast<LPCVOID>(reinterpret_cast<UINT_PTR>(MemoryContext.m_Source)     + ConsumedBytes);
			MemoryContext.m_Destination = reinterpret_cast<LPVOID>(reinterpret_cast<UINT_PTR>(MemoryContext.m_Destination) + ConsumedBytes);
			MemoryContext.m_Bytes		-= ConsumedBytes;

			return;

		default:

			Sequential													 = reinterpret_cast<CONST T*>(MemoryContext.m_Source)[Iteration];
			
			reinterpret_cast<T*>(MemoryContext.m_Destination)[Iteration] = Sequential;

			Iteration++;

		}
	}

}

class MEMORY_COPY {
public:

	static CONST MEMORY_COPY& Instance() noexcept {
		static MEMORY_COPY Object;
		return Object;
	}

	VOID Copy(
		LPCVOID Source,
		LPVOID  Destination,
		SIZE_T	Bytes
	) CONST noexcept {
						  
		MEMORY_COPY_CONTEXT	MemoryContext(Source, Destination, Bytes);

		switch (this->m_StartingMemoryCopyIndex) {
		case static_cast<UINT64>(PROCESSOR_FEATURE::AVX512):
			this->MemoryCopyAVX512(MemoryContext);
			[[fallthrough]];
		case static_cast<UINT64>(PROCESSOR_FEATURE::AVX2):
			this->MemoryCopyAVX2  (MemoryContext);
			[[fallthrough]];
		case static_cast<UINT64>(PROCESSOR_FEATURE::SSE2):
			this->MemoryCopySSE2  (MemoryContext);
			[[fallthrough]];
		case static_cast<UINT64>(PROCESSOR_FEATURE::SEQUENTIAL):
			this->MemoryCopyUINT64(MemoryContext);
			this->MemoryCopyUINT32(MemoryContext);
			this->MemoryCopyUINT16(MemoryContext);
			this->MemoryCopyUINT8 (MemoryContext);
			return;
		}

	}

private:

	/*
		If the program does not allow for AVX512 instantiation, remove the ternary operator for its feature. Do not change the enumeration values. Leave those the same. 
		PROCESSOR_FEATURE::AVX512 should still equal 0 (and be included) even if the AVX512 instantiations will be removed (not by hand, but by the compiler).

		For example, not wanting the AVX512 control paths in the program (say that the lowest common denominator is AVX2 support),
		adjust the MEMORY_COPY constructor like so:

		m_StartingMemoryCopyIndex((IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE))	? static_cast<UINT64>(PROCESSOR_FEATURE::AVX2)       :
								  (IsProcessorFeaturePresent(PF_AVX_INSTRUCTIONS_AVAILABLE))	? static_cast<UINT64>(PROCESSOR_FEATURE::SSE2)       :
																								  static_cast<UINT64>(PROCESSOR_FEATURE::SEQUENTIAL)
		) {}


		Do not try to add the removed vectorized control path back by runtime support, the compiler cannot see this information.
		The user address space on Windows (as of November 25, 2026) is 128 terabytes.

		Even if using same function signature and code, the vectorized control paths seen by the compiler will still be faster through throughput and throughout all iterations
		than a runtime vectorized control path.

	*/


	/*
		Below are helper macros. During development of a program, programmers do not need to define any of these macros. 
		The CopyMemory procedure will use the highest denominator processor feature. When it comes time to release a program to users,
		define the programs's lowest common denominator processor feature macro. The higher processor feature(s) will not be compiled, avoiding a EVENT 6 #UD fault.

		Without the macros below, the processor, OS, and compiler (MSVC) will not allow modification of control registers. If compiled on a processor that supports AVX512F,
		the templated functions will be instantiated, an EVENT 6 #UD fault is inevitable when CopyMemory procedure is called, when executable is released and 
		and it is launched on unsupported CPUs.

		The helper macros below are needed in the above scenario.

	*/

#if defined(PROCESSOR_FEATURE_AVX512F)
	
	MEMORY_COPY() noexcept : m_StartingMemoryCopyIndex((IsProcessorFeaturePresent(PF_AVX512F_INSTRUCTIONS_AVAILABLE)) ? static_cast<UINT64>(PROCESSOR_FEATURE::AVX512)     :
													   (IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE))    ? static_cast<UINT64>(PROCESSOR_FEATURE::AVX2)       :
													   (IsProcessorFeaturePresent(PF_XMMI64_INSTRUCTIONS_AVAILABLE))  ? static_cast<UINT64>(PROCESSOR_FEATURE::SSE2)       :
																														static_cast<UINT64>(PROCESSOR_FEATURE::SEQUENTIAL)
													  ) {}
#elif defined(PROCESSOR_FEATURE_AVX2)
	
	MEMORY_COPY() noexcept : m_StartingMemoryCopyIndex((IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE))    ? static_cast<UINT64>(PROCESSOR_FEATURE::AVX2)	   :
													   (IsProcessorFeaturePresent(PF_XMMI64_INSTRUCTIONS_AVAILABLE))  ? static_cast<UINT64>(PROCESSOR_FEATURE::SSE2)	   :
																													    static_cast<UINT64>(PROCESSOR_FEATURE::SEQUENTIAL)
													  ) {}
														
#elif defined(PROCESSOR_FEATURE_SSE2)

	MEMORY_COPY() noexcept : m_StartingMemoryCopyIndex((IsProcessorFeaturePresent(PF_XMMI64_INSTRUCTIONS_AVAILABLE)) ? static_cast<UINT64>(PROCESSOR_FEATURE::SSE2)		   :
																													   static_cast<UINT64>(PROCESSOR_FEATURE::SEQUENTIAL)
													  ) {}

#elif defined(PROCESSOR_FEATURE_SEQUENTIAL)

	MEMORY_COPY() noexcept : m_StartingMemoryCopyIndex(static_cast<UINT64>(PROCESSOR_FEATURE::SEQUENTIAL)) {}

#else
	
	MEMORY_COPY() noexcept : m_StartingMemoryCopyIndex((IsProcessorFeaturePresent(PF_AVX512F_INSTRUCTIONS_AVAILABLE)) ? static_cast<UINT64>(PROCESSOR_FEATURE::AVX512)     :
													   (IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE))    ? static_cast<UINT64>(PROCESSOR_FEATURE::AVX2)	   :
													   (IsProcessorFeaturePresent(PF_XMMI64_INSTRUCTIONS_AVAILABLE))  ? static_cast<UINT64>(PROCESSOR_FEATURE::SSE2)	   :
																														static_cast<UINT64>(PROCESSOR_FEATURE::SEQUENTIAL)
													  ) {}
#endif


	VOID MemoryCopyAVX512(
		MEMORY_COPY_CONTEXT& MemoryContext
	) CONST noexcept {

		MemoryCopyVectorized<__m512i>(MemoryContext);

	}

	VOID MemoryCopyAVX2(
		MEMORY_COPY_CONTEXT& MemoryContext
	) CONST noexcept {

		MemoryCopyVectorized<__m256i>(MemoryContext);
	}

	VOID MemoryCopySSE2(
		MEMORY_COPY_CONTEXT& MemoryContext
	) CONST noexcept {

		MemoryCopyVectorized<__m128i>(MemoryContext);

	}

	VOID MemoryCopyUINT64(
		MEMORY_COPY_CONTEXT& MemoryContext
	) CONST noexcept {
		MemoryCopySequential<UINT64>(MemoryContext);
	}

	VOID MemoryCopyUINT32(
		MEMORY_COPY_CONTEXT& MemoryContext
	) CONST noexcept {
		MemoryCopySequential<UINT32>(MemoryContext);
	}

	VOID MemoryCopyUINT16(
		MEMORY_COPY_CONTEXT& MemoryContext
	) CONST noexcept {
		MemoryCopySequential<UINT16>(MemoryContext);
	}

	VOID MemoryCopyUINT8(
		MEMORY_COPY_CONTEXT& MemoryContext
	) CONST noexcept {
		MemoryCopySequential<UINT8>(MemoryContext);
	}

	CONST UINT64 m_StartingMemoryCopyIndex;
	
};

ANONYMOUS_NAMESPACE_END


VOID MemoryCopy(
	LPCVOID    Source,
	LPVOID     Destination,
	SIZE_T	   Bytes
) {

	assert(reinterpret_cast<UINT_PTR>(Source) - NULL);
	assert(reinterpret_cast<UINT_PTR>(Destination) - NULL);
	assert(Bytes - 0);

	MEMORY_COPY::Instance().Copy(Source, Destination, Bytes);

	return;

}
