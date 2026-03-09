from abc import ABC, abstractmethod

class ShellMaterial(ABC):
  @abstractmethod
  def membrane_stress(self, eps):
    pass

  @abstractmethod
  def bending_stress(self, kappa):
    pass

  @abstractmethod
  def shear_stress(self, gamma):
    pass