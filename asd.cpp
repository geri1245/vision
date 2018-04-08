#include <vector>
#include <iostream>
#include <algorithm>

int main()
{
  std::vector<int> a, b;
  a.reserve(10);
  b.reserve(10);

  for(int i = 0; i < 10; ++i)
  {
    a.push_back(i);
  }

  std::transform(a.begin(), a.end(), b.begin(), [](int i){ return 2 * i; });

  std::cout << a.size() << " " << b.size() << "\n";

  return 0;
}
