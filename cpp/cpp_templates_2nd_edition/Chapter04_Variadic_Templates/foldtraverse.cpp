#include <iostream>

struct Node {
  int value;
  Node *left;
  Node *right;
  Node(int i = 0) : value(i), left(nullptr), right(nullptr) {}
};

auto left = &Node::left;
auto right = &Node::right;

template <typename T, typename... TP> Node *traverse(T np, TP... paths) {
  return (np->*...->*paths);
}

int main() {
  Node *root = new Node{0};
  root->left = new Node{1};
  root->left->right = new Node{3};
  root->left->right->left = new Node{4};
  root->left->right->right = new Node{5};

  Node *node = traverse(root, left, right, right);
  if (node) {
    std::cout << node->value << std::endl;
  }
  return 0;
}
