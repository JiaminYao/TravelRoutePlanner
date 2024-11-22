# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 `pkg-config --cflags --libs opencv4`

# Targets and sources
TARGETS = E1 E2 E3
SOURCES = $(addsuffix .cpp, $(TARGETS))

# Build targets
all: $(TARGETS)

E1: E1.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

E2: E2.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

E3: E3.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Clean up build artifacts
clean:
	rm -f $(TARGETS)
