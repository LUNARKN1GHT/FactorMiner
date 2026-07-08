from typing import Tuple, Optional
import gymnasium as gym
import math

from alphagen.config import MAX_EXPR_LENGTH
from alphagen.data.exception import InvalidExpressionException
from alphagen.data.tokens import *
from alphagen.data.expression import *
from alphagen.data.tree import ExpressionBuilder
from alphagen.models.alpha_pool import AlphaPoolBase
from alphagen.models.linear_alpha_pool import LinearAlphaPool
from alphagen.utils import reseed_everything


class AlphaEnvCore(gym.Env):
    pool: AlphaPoolBase
    _tokens: List[Token]
    _builder: ExpressionBuilder
    _print_expr: bool

    def __init__(
        self,
        pool: AlphaPoolBase,
        device: torch.device = torch.device('cuda:0'),
        print_expr: bool = False
    ):
        super().__init__()

        self.pool = pool
        self._print_expr = print_expr
        self._device = device

        self.eval_cnt = 0

        self.render_mode = None
        self.reset()

    def reset(
        self, *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Tuple[List[Token], dict]:
        reseed_everything(seed)
        self._tokens = [BEG_TOKEN]
        self._builder = ExpressionBuilder()
        return self._tokens, self._valid_action_types()

    def step(self, action: Token) -> Tuple[List[Token], float, bool, bool, dict]:
        if (isinstance(action, SequenceIndicatorToken) and
                action.indicator == SequenceIndicatorType.SEP):
            reward = self._evaluate()
            done = True
        elif len(self._tokens) < MAX_EXPR_LENGTH:
            self._tokens.append(action)
            try:
                self._builder.add_token(action)
                done = False
                reward = 0.0
            except InvalidExpressionException:
                # 动作掩码（action_masks，见 wrapper.py）跟 ExpressionBuilder.validate() 之间
                # 存在没查清根因的边界不一致，长跑偶尔会选到实际非法的 token 导致这里抛异常，
                # 崩掉整个训练进程。这里防御性兜底：当成本回合失败（跟 MAX_EXPR_LENGTH 时
                # is_valid()==False 的处理方式一致，reward=-1），不让偶发边界 case 打断长跑，
                # 不改变合法路径下的任何行为。
                done = True
                reward = -1.
        else:
            done = True
            reward = self._evaluate() if self._builder.is_valid() else -1.

        if math.isnan(reward):
            reward = 0.

        return self._tokens, reward, done, False, self._valid_action_types()

    def _evaluate(self):
        expr: Expression = self._builder.get_tree()
        if self._print_expr:
            print(expr)
        try:
            ret = self.pool.try_new_expr(expr)
            self.eval_cnt += 1
            return ret
        except OutOfDataRangeError:
            return 0.

    def _valid_action_types(self) -> dict:
        valid_op_unary = self._builder.validate_op(UnaryOperator)
        valid_op_binary = self._builder.validate_op(BinaryOperator)
        valid_op_rolling = self._builder.validate_op(RollingOperator)
        valid_op_pair_rolling = self._builder.validate_op(PairRollingOperator)

        valid_op = valid_op_unary or valid_op_binary or valid_op_rolling or valid_op_pair_rolling
        valid_dt = self._builder.validate_dt()
        valid_const = self._builder.validate_const()
        valid_feature = self._builder.validate_featured_expr()
        valid_stop = self._builder.is_valid()

        ret = {
            'select': [valid_op, valid_feature, valid_const, valid_dt, valid_stop],
            'op': {
                UnaryOperator: valid_op_unary,
                BinaryOperator: valid_op_binary,
                RollingOperator: valid_op_rolling,
                PairRollingOperator: valid_op_pair_rolling
            }
        }
        return ret

    def valid_action_types(self) -> dict:
        return self._valid_action_types()

    def render(self, mode='human'):
        pass
