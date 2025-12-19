# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 `pkg-config --cflags --libs opencv4`

# Targets and sources
TARGETS = E1
SOURCES = $(addsuffix .cpp, $(TARGETS))

# Build targets
E1: E1.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Clean up build artifacts
clean:
	rm -f $(TARGETS)
