#pragma once
#include <optional>
#include <vector>


class NumberSource{
    public:
        virtual std::optional<int> nextNum() = 0;
        virtual ~NumberSource() = default;
};

struct CounterSource:NumberSource{
    public:
    std::optional<int> nextNum() override{
        if(currNum > limit) return std::nullopt;
        else return currNum++;
    }
    private:
    int currNum = 0;
    int limit = 10;
};

struct SequenceSource:NumberSource{
    public:
    std::optional<int> nextNum() override{
        if(idx >= numbers.size()) return std::nullopt;
        
        int curr = numbers[idx];
        idx++;
        return curr;
    }

    void addNum(int n){ numbers.push_back(n);}
    private:
    std::vector<int> numbers;
    int idx = 0;
};